
"""
    绘图工具模块

    该模块提供了用于可视化训练过程中loss和accuracy的图表功能。
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # 添加这个导入


class PlotManager:
    """用于管理训练过程中的loss和accuracy数据，在每个epoch更新同一张图表"""

    def __init__(self, save_dir='plots'):
        """初始化绘图管理器

        Args:
            save_dir: 图表保存目录
        """
        self.train_losses = []  # 训练损失记录
        self.valid_accuracies = []  # 验证准确率记录
        self.epochs = []  # 轮次记录
        self.num_epochs = 0  # 总轮次记录
        self.save_dir = save_dir
        
        # 记录训练开始时间戳
        self.start_time = datetime.now()
        self.timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        
        # 跟踪最佳指标
        self.min_loss = float('inf')
        self.max_accuracy = 0.0
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        print(f"图表将保存至目录: {save_dir}")
        print(f"训练开始时间戳: {self.timestamp}")
        
        # 设置图表文件名（使用训练开始时间戳）
        self.filename = f'training_metrics_{self.timestamp}.png'
        self.save_path = os.path.join(self.save_dir, self.filename)

    def update(self, epoch, train_loss, valid_accuracy, num_epochs):
        """更新训练数据并更新图表

        Args:
            epoch: 当前轮次
            train_loss: 当前轮次的训练损失
            valid_accuracy: 当前轮次的验证准确率
            num_epochs: 总轮次
        """
        # 记录数据
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.valid_accuracies.append(valid_accuracy)
        self.num_epochs = num_epochs
        
        # 更新最佳指标
        if train_loss < self.min_loss:
            self.min_loss = train_loss
        if valid_accuracy > self.max_accuracy:
            self.max_accuracy = valid_accuracy
        
        # 每个epoch都更新并保存同一张图表
        self._update_plot()

    def _update_plot(self):
        """更新并保存图表"""
        try:
            # 设置中文字体支持
            plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]
            plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
            
            # 创建图表和子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            fig.subplots_adjust(wspace=0.3)  # 调整子图间距
            
            # 设置图表标题
            fig.suptitle(f'训练过程监控 - 开始时间: {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}\n当前轮次: {self.epochs[-1]}/{self.num_epochs}', fontsize=16)
            
            # 绘制训练损失曲线
            ax1.plot(self.epochs, self.train_losses, 'b-', linewidth=2, label='训练损失')
            ax1.set_xlabel('轮次')
            ax1.set_ylabel('损失值')
            ax1.set_title(f'训练损失曲线 - 平均损失: {self.min_loss:.4f}')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(loc='best')
            # 设置X轴为整数刻度
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # 绘制验证准确率曲线
            ax2.plot(self.epochs, self.valid_accuracies, 'r-', linewidth=2, label='验证准确率')
            ax2.set_xlabel('轮次')
            ax2.set_ylabel('准确率')
            ax2.set_title(f'验证准确率曲线 - 准确率: {self.max_accuracy:.4f}')
            ax2.set_ylim(0, 1.05)  # 准确率范围0-100%
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(loc='best')
            # 设置X轴为整数刻度
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # 添加每隔5个epoch的数据点标注
            for i, epoch in enumerate(self.epochs):
                # 第0个epoch也显示，然后每隔5个epoch显示一次
                if epoch % 5 == 0 or epoch == 0:
                    # 标注训练损失
                    ax1.annotate(f'{self.train_losses[i]:.4f}', 
                                xy=(epoch, self.train_losses[i]),
                                xytext=(0, 5),
                                textcoords='offset points',
                                ha='center',
                                fontsize=8,
                                color='blue',
                                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                    
                    # 标注验证准确率
                    ax2.annotate(f'{self.valid_accuracies[i]:.4f}', 
                                xy=(epoch, self.valid_accuracies[i]),
                                xytext=(0, 5),
                                textcoords='offset points',
                                ha='center',
                                fontsize=8,
                                color='red',
                                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
            
            # 保存图表（覆盖已有文件）
            fig.savefig(self.save_path, dpi=300, bbox_inches='tight')
            print(f"epoch {self.epochs[-1]} 图表已更新至: {self.save_path}")
            
            # 清理资源
            plt.close(fig)
        except Exception as e:
            print(f"更新epoch {self.epochs[-1]} 图表时出错: {e}")

    def save_plot(self, filename=None):
        """保存最终的完整训练过程图表

        Args:
            filename: 保存的文件名，如果为None则使用已设置的文件名
        """
        # 如果未提供文件名，则使用已设置的文件名（训练开始时间戳）
        if filename is not None:
            self.filename = filename
            self.save_path = os.path.join(self.save_dir, self.filename)
        
        # 确保有数据可以绘图
        if not self.epochs:
            print("没有训练数据可以生成图表")
            return
        
        try:
            # 再次调用更新方法，确保最终图表是最新的
            self._update_plot()
            print(f"最终图表已保存至: {self.save_path}")
        except Exception as e:
            print(f"保存最终图表时出错: {e}")

    def close(self):
        """清理资源"""
        print("绘图管理器已关闭")