from typing import Dict, List, Optional, Set
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from datetime import datetime
from config import config
from pathlib import Path
import psutil
import os

logger = logging.getLogger(__name__)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    logger.info(f"内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")

class CodeEmbeddingsService:
    def __init__(self):
        """初始化代码向量化服务"""
        logger.info("初始化代码向量化服务...")
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型和分词器
        try:
            logger.info(f"加载模型: {config.EMBEDDING_MODEL}")
            self.tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
            self.model = AutoModel.from_pretrained(config.EMBEDDING_MODEL).to(self.device)
            self.model.eval()  # 设置为评估模式
            logger.info("模型加载完成")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

        # 初始化向量存储
        self.collection_name = config.VECTOR_COLLECTION
        logger.info(f"正在连接向量数据库: {config.VECTOR_DB_PATH}")
        self.qdrant = QdrantClient(path=str(config.VECTOR_DB_PATH))
        logger.info("向量数据库连接成功")
        self._init_collection()
        
    def _init_collection(self):
        """初始化或获取向量集合"""
        vector_size = self.model.config.hidden_size
        
        try:
            # 检查向量数据库目录权限
            db_path = Path(config.VECTOR_DB_PATH)
            if not db_path.exists():
                logger.info(f"创建向量数据库目录: {db_path}")
                db_path.mkdir(parents=True, exist_ok=True)
            
            # 检查集合是否存在
            try:
                self.qdrant.get_collection(self.collection_name)
                logger.info(f"已连接到现有集合: {self.collection_name}")
            except Exception as e:
                logger.info(f"创建新的向量集合: {self.collection_name}")
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                        on_disk=True  # 使用磁盘存储以节省内存
                    )
                )
                logger.info("向量集合创建成功")
                
        except Exception as e:
            logger.error(f"初始化向量集合时出错: {str(e)}", exc_info=True)
            raise
        
    def compute_embeddings(self, code: str) -> np.ndarray:
        """计算代码的向量表示"""
        try:
            # 对输入进行分词
            inputs = self.tokenizer(
                code,
                padding=True,
                truncation=True,
                max_length=512,  # 限制最大长度
                return_tensors="pt"
            ).to(self.device)
            
            # 计算向量表示
            with torch.no_grad():  # 不计算梯度
                outputs = self.model(**inputs)
                # 使用最后一层的 [CLS] token 作为向量表示
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"计算向量表示失败: {str(e)}")
            raise

    def _get_stored_files_info(self) -> Dict[str, datetime]:
        """获取已存储文件的信息"""
        try:
            stored_points = self.qdrant.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False
            )[0]
            return {
                point.payload['file_path']: datetime.fromisoformat(point.payload['last_updated'])
                for point in stored_points
            }
        except Exception:
            return {}

    def build_index(self, code_files: Dict[str, str]):
        """构建代码文件的向量索引"""
        log_memory_usage()
        logger.info(f"检查文件更新，总文件数: {len(code_files)}")
        
        # 获取已存储的文件信息
        stored_files = self._get_stored_files_info()
        
        # 找出需要更新的文件
        files_to_process = {}
        for file_path, content in code_files.items():
            content_hash = self._compute_content_hash(content)
            
            # 检查文件是否存在或内容是否变化
            if file_path not in stored_files:
                files_to_process[file_path] = content
                logger.debug(f"新文件: {file_path}")
            else:
                # 查找现有文件的哈希值
                scroll_result = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=file_path)
                        )]
                    ),
                    with_payload=True,
                    limit=1
                )
                points = scroll_result[0]
                
                if points and points[0].payload.get('content_hash') != content_hash:
                    files_to_process[file_path] = content
                    logger.debug(f"文件内容已变化: {file_path}")
        
        if not files_to_process:
            logger.info("所有文件都已是最新状态，无需更新向量")
            return
            
        logger.info(f"开始处理 {len(files_to_process)} 个新文件")
        
        # 批量处理向量
        batch_size = config.BATCH_SIZE
        total_files = len(files_to_process)
        processed_files = 0
        points = []
        
        for file_path, content in files_to_process.items():
            try:
                embedding = self.compute_embeddings(content)[0]
                
                point = models.PointStruct(
                    id=self._get_next_point_id(),
                    vector=embedding.tolist(),
                    payload={
                        'file_path': file_path,
                        'last_updated': datetime.now().isoformat(),
                        'file_size': len(content),
                        'content_hash': self._compute_content_hash(content)
                    }
                )
                points.append(point)
                processed_files += 1
                
                if len(points) >= batch_size:
                    logger.info(f"上传批次向量，进度: {processed_files}/{total_files}")
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    points = []
                
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
                continue
        
        # 上传剩余的点
        if points:
            logger.info(f"上传最后一批向量，进度: {processed_files}/{total_files}")
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        logger.info(f"索引更新完成，处理了 {processed_files} 个新文件")

    def _compute_content_hash(self, content: str) -> str:
        """计算内容的哈希值"""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()

    def _get_next_point_id(self) -> int:
        """获取下一个可用的点ID"""
        try:
            collection_info = self.qdrant.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception:
            return 0

    def find_related_files(self, query_code: str, max_files: int = 3) -> List[str]:
        """查找与给定代码最相关的文件"""
        try:
            # 计算查询代码的向量
            query_embedding = self.compute_embeddings(query_code)[0]
            
            # 使用 Qdrant 搜索相似向量
            search_result = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=max_files
            )
            
            # 返回相关文件路径
            return [hit.payload['file_path'] for hit in search_result]
            
        except Exception as e:
            logger.error(f"查找相关文件失败: {str(e)}")
            return []
    
    def update_file(self, file_path: str, content: str):
        """更新文件的向量表示"""
        try:
            logger.debug(f"更新文件向量: {file_path}")
            embedding = self.compute_embeddings(content)[0]
            
            # 使用 Qdrant 存储向量
            point = models.PointStruct(
                id=self._get_next_point_id(),
                vector=embedding.tolist(),
                payload={
                    'file_path': file_path,
                    'last_updated': datetime.now().isoformat(),
                    'file_size': len(content),
                    'content_hash': self._compute_content_hash(content)
                }
            )
            
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"文件向量更新成功: {file_path}")
            
        except Exception as e:
            logger.error(f"更新文件向量失败 {file_path}: {str(e)}")
            raise

    def cleanup_old_vectors(self, active_files: Set[str]):
        """清理不存在的文件的向量"""
        try:
            # 获取所有存储的文件路径
            stored_files = self._get_stored_files_info()
            
            # 找出需要删除的文件
            files_to_remove = set(stored_files.keys()) - active_files
            
            if files_to_remove:
                # 获取要删除的点的ID
                points_to_remove = []
                for file_path in files_to_remove:
                    scroll_result = self.qdrant.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=models.Filter(
                            must=[models.FieldCondition(
                                key="file_path",
                                match=models.MatchValue(value=file_path)
                            )]
                        ),
                        with_payload=False,
                        limit=1
                    )
                    if scroll_result[0]:
                        points_to_remove.append(scroll_result[0][0].id)
                
                # 批量删除点
                if points_to_remove:
                    self.qdrant.delete(
                        collection_name=self.collection_name,
                        points_selector=models.PointIdsList(
                            points=points_to_remove
                        )
                    )
                    logger.info(f"清理了 {len(points_to_remove)} 个旧向量")
                
        except Exception as e:
            logger.error(f"清理旧向量时出错: {str(e)}") 