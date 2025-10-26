import datetime
import json
import os
import platform
from datetime import datetime  # 注意这里的导入方式
from typing import Dict, Any, List

import toml
import torch


class TrainingLogger:
    """训练日志记录器，用于保存每次训练的超参数和核心信息"""
    
    def __init__(self, log_dir: str = "training_logs"):
        """初始化日志记录器
        
        Args:    
            log_dir: 日志文件保存目录
        """
        self.base_log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 修复这里，直接使用 datetime.now()
        self.log_dir = os.path.join(self.base_log_dir, self.timestamp)  # 在training_logs下创建时间戳子目录
        self.log_data = {
            "training_info": {},
            "hyperparameters": {},
            "system_info": {},
            "additional_info": {}
        }
        
        # 创建日志目录（如果不存在）
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 记录系统信息
        self.log_data["system_info"] = {
            "timestamp": self.timestamp,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 修复这里
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__ if torch.cuda.is_available() else "Not available",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        }
    
    def load_config(self, config_path: str) -> None:
        """从配置文件加载超参数
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.toml'):
                    config = toml.load(f)
                elif config_path.endswith('.json'):
                    config = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path}")
            
            self.log_data["hyperparameters"] = config
            self.log_data["training_info"]["config_path"] = config_path
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    def add_hyperparameters(self, params: Dict[str, Any]) -> None:
        """添加额外的超参数"""
        self.log_data["hyperparameters"].update(params)
    
    def add_training_info(self, info: Dict[str, Any]) -> None:
        """添加训练相关信息"""
        self.log_data["training_info"].update(info)
    
    def add_additional_info(self, info: Dict[str, Any]) -> None:
        """添加其他额外信息"""
        self.log_data["additional_info"].update(info)
    
    def save_log(self, suffix: str = "") -> str:
        """保存日志到文件"""
        # 更新结束时间
        self.log_data["system_info"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 修复这里
        
        # 构建文件名
        filename = f"training_params"
        if suffix:
            filename = f"{filename}_{suffix}"
        filename = f"{filename}.json"
        
        # 保存日志到文件
        log_path = os.path.join(self.log_dir, filename)
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(self.log_data, f, ensure_ascii=False, indent=2)
            print(f"训练参数日志已保存到: {log_path}")
            return log_path
        except Exception as e:
            print(f"保存训练参数日志失败: {e}")
            return ""
    
    def get_log_data(self) -> Dict[str, Any]:
        """获取日志数据"""
        return self.log_data
    
    def get_timestamp(self) -> str:
        """获取当前日志的时间戳"""
        return self.timestamp


class TrainingProcessLogger:
    """训练过程记录器，专门用于记录训练过程中的详细信息"""
    
    def __init__(self, log_dir: str = "training_logs", timestamp: str = None):
        """初始化训练过程记录器"""
        self.base_log_dir = log_dir
        self.timestamp = timestamp if timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(self.base_log_dir, self.timestamp)  # 在training_logs下创建时间戳子目录
        self.log_file_path = os.path.join(self.log_dir, "training_process.log")
        
        # 创建日志目录（如果不存在）
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化日志文件（清空内容）
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"===== 训练过程日志开始 =====\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write(f"日志文件: {self.log_file_path}\n")
            f.write("=========================\n\n")
    
    def log_training_batch(self, epoch: int, batch: int, total_batches: int, loss: float) -> None:
        """记录训练批次信息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"{timestamp} [TRAIN] epoch={epoch}, batch={batch}/{total_batches}, loss={loss:.5f}\n"
        
        # 直接写入日志文件，不打印到控制台
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def log_validation_batch(self, epoch: int, batch: int, total_batches: int) -> None:
        """记录验证批次信息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"{timestamp} [VALID] epoch={epoch}, batch={batch}/{total_batches}\n"
        
        # 直接写入日志文件，不打印到控制台
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def log_epoch_end(self, epoch: int, avg_loss: float, accuracy: float, lr: float, best_accuracy: float = None) -> None:
        """记录epoch结束信息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 构建基础日志条目
        log_entry = f"{timestamp} [EPOCH_END] epoch={epoch}, avg_loss={avg_loss:.5f}, accuracy={accuracy:.4f}, lr={lr:.8f}"
        
        # 如果有best_accuracy，添加到日志
        if best_accuracy is not None:
            log_entry += f", best_accuracy={best_accuracy:.4f}"
        log_entry += "\n"
        
        # 直接写入日志文件，不打印到控制台
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def log_model_saving(self, epoch: int, path: str, is_best: bool = False) -> None:
        """记录模型保存信息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        model_type = "best" if is_best else "last"
        log_entry = f"{timestamp} [MODEL_SAVE] epoch={epoch}, type={model_type}, path={path}\n"
        
        # 直接写入日志文件，不打印到控制台
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def log_training_event(self, event_type: str, message: str) -> None:
        """记录训练中的事件"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"{timestamp} [EVENT] type={event_type}, message={message}\n"
        
        # 直接写入日志文件，不打印到控制台
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def save_log(self) -> str:
        """标记日志结束并返回日志文件路径"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 写入日志结束标记
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write("\n=========================\n")
            f.write(f"===== 训练过程日志结束 =====\n")
            f.write(f"结束时间: {timestamp}\n")
        
        # 只返回日志路径，不打印到控制台
        return self.log_file_path


# 便捷函数：一键保存配置文件和训练信息
def save_training_log(config_path: str, 
                      additional_hparams: Dict[str, Any] = None, 
                      training_info: Dict[str, Any] = None, 
                      log_dir: str = "training_logs") -> str:
    """便捷函数：保存训练日志
    
    Args:
        config_path: 配置文件路径
        additional_hparams: 额外的超参数
        training_info: 训练信息
        log_dir: 日志保存目录
        
    Returns:
        str: 保存的日志文件路径
    """
    logger = TrainingLogger(log_dir)
    logger.load_config(config_path)
    
    if additional_hparams:
        logger.add_hyperparameters(additional_hparams)
    
    if training_info:
        logger.add_training_info(training_info)
    
    return logger.save_log()


# 示例使用方法
if __name__ == "__main__":
    # 使用便捷函数快速保存日志
    config_file = "../configs/config_OneGPU.toml"  # 根据实际路径调整
    if os.path.exists(config_file):
        extra_params = {"experiment_name": "test_experiment"}
        train_info = {"dataset_size": {"train": 1000, "valid": 200}}
        log_path = save_training_log(config_file, extra_params, train_info)
        print(f"日志已保存到: {log_path}")