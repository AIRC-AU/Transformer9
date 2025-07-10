#!/usr/bin/env python3
"""
知识蒸馏过程可视化工具
Visualization tool for knowledge distillation process
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class DistillationVisualizer:
    """知识蒸馏可视化器"""
    
    def __init__(self, figsize=(16, 12)):
        self.figsize = figsize
        self.colors = {
            'teacher': '#FF6B6B',      # 红色 - 教师模型
            'student': '#4ECDC4',      # 青色 - 学生模型
            'distillation': '#45B7D1', # 蓝色 - 蒸馏过程
            'data': '#96CEB4',         # 绿色 - 数据流
            'loss': '#FFEAA7',         # 黄色 - 损失函数
            'output': '#DDA0DD'        # 紫色 - 输出
        }
    
    def create_comprehensive_diagram(self):
        """创建综合的知识蒸馏流程图"""
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 15)
        ax.axis('off')
        
        # 标题
        ax.text(10, 14, 'Transformer知识蒸馏完整流程', 
                fontsize=20, fontweight='bold', ha='center')
        
        # 1. 数据输入部分
        self._draw_data_input(ax)
        
        # 2. 教师模型部分
        self._draw_teacher_model(ax)
        
        # 3. 学生模型部分
        self._draw_student_model(ax)
        
        # 4. 蒸馏损失部分
        self._draw_distillation_loss(ax)
        
        # 5. 连接线和数据流
        self._draw_connections(ax)
        
        # 6. 图例
        self._draw_legend(ax)
        
        plt.tight_layout()
        return fig
    
    def _draw_data_input(self, ax):
        """绘制数据输入部分"""
        # 输入数据框
        input_box = FancyBboxPatch(
            (1, 11), 3, 1.5,
            boxstyle="round,pad=0.1",
            facecolor=self.colors['data'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(input_box)
        ax.text(2.5, 11.75, '输入数据\n(源语言句子)', 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 目标数据框
        target_box = FancyBboxPatch(
            (1, 9), 3, 1.5,
            boxstyle="round,pad=0.1",
            facecolor=self.colors['data'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(target_box)
        ax.text(2.5, 9.75, '目标数据\n(目标语言句子)', 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    def _draw_teacher_model(self, ax):
        """绘制教师模型部分"""
        # 教师模型主体
        teacher_box = FancyBboxPatch(
            (6, 10), 4, 3,
            boxstyle="round,pad=0.1",
            facecolor=self.colors['teacher'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(teacher_box)
        
        # 教师模型内部结构
        ax.text(8, 12.5, '教师模型 (Teacher)', 
                ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(8, 12, '• 12层Transformer', 
                ha='center', va='center', fontsize=9)
        ax.text(8, 11.6, '• d_model=512', 
                ha='center', va='center', fontsize=9)
        ax.text(8, 11.2, '• 8个注意力头', 
                ha='center', va='center', fontsize=9)
        ax.text(8, 10.8, '• 44M参数', 
                ha='center', va='center', fontsize=9)
        ax.text(8, 10.4, '• 预训练完成', 
                ha='center', va='center', fontsize=9, style='italic')
        
        # 教师输出
        teacher_output = FancyBboxPatch(
            (11.5, 10.5), 2.5, 2,
            boxstyle="round,pad=0.1",
            facecolor=self.colors['output'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(teacher_output)
        ax.text(12.75, 11.5, '教师输出\nLogits', 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    def _draw_student_model(self, ax):
        """绘制学生模型部分"""
        # 学生模型主体
        student_box = FancyBboxPatch(
            (6, 6), 4, 3,
            boxstyle="round,pad=0.1",
            facecolor=self.colors['student'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(student_box)
        
        # 学生模型内部结构
        ax.text(8, 8.5, '学生模型 (Student)', 
                ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(8, 8, '• 6层Transformer', 
                ha='center', va='center', fontsize=9)
        ax.text(8, 7.6, '• d_model=256', 
                ha='center', va='center', fontsize=9)
        ax.text(8, 7.2, '• 4个注意力头', 
                ha='center', va='center', fontsize=9)
        ax.text(8, 6.8, '• 10M参数', 
                ha='center', va='center', fontsize=9)
        ax.text(8, 6.4, '• 正在训练', 
                ha='center', va='center', fontsize=9, style='italic')
        
        # 学生输出
        student_output = FancyBboxPatch(
            (11.5, 6.5), 2.5, 2,
            boxstyle="round,pad=0.1",
            facecolor=self.colors['output'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(student_output)
        ax.text(12.75, 7.5, '学生输出\nLogits', 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    def _draw_distillation_loss(self, ax):
        """绘制蒸馏损失部分"""
        # 温度缩放
        temp_box = FancyBboxPatch(
            (15, 10), 2.5, 1.5,
            boxstyle="round,pad=0.1",
            facecolor=self.colors['distillation'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(temp_box)
        ax.text(16.25, 10.75, '温度缩放\nT=4.0', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # KL散度损失
        kl_box = FancyBboxPatch(
            (15, 8), 2.5, 1.5,
            boxstyle="round,pad=0.1",
            facecolor=self.colors['loss'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(kl_box)
        ax.text(16.25, 8.75, 'KL散度损失\nα=0.8', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # 交叉熵损失
        ce_box = FancyBboxPatch(
            (15, 6), 2.5, 1.5,
            boxstyle="round,pad=0.1",
            facecolor=self.colors['loss'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(ce_box)
        ax.text(16.25, 6.75, '交叉熵损失\nβ=0.2', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # 总损失
        total_loss_box = FancyBboxPatch(
            (15, 3.5), 2.5, 1.5,
            boxstyle="round,pad=0.1",
            facecolor='#FF9999',
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(total_loss_box)
        ax.text(16.25, 4.25, '总损失\nL_total', 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 反向传播箭头
        ax.annotate('', xy=(8, 6), xytext=(16.25, 3.5),
                   arrowprops=dict(arrowstyle='->', lw=3, color='red'))
        ax.text(12, 4.5, '反向传播\n梯度更新', 
                ha='center', va='center', fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def _draw_connections(self, ax):
        """绘制连接线和数据流"""
        # 输入到模型的连接
        ax.annotate('', xy=(6, 11.5), xytext=(4, 11.75),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        ax.annotate('', xy=(6, 7.5), xytext=(4, 9.75),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        
        # 模型到输出的连接
        ax.annotate('', xy=(11.5, 11.5), xytext=(10, 11.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
        ax.annotate('', xy=(11.5, 7.5), xytext=(10, 7.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
        
        # 输出到损失的连接
        ax.annotate('', xy=(15, 10.75), xytext=(14, 11.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
        ax.annotate('', xy=(15, 8.75), xytext=(14, 11.5),
                   arrowprops=dict(arrowstyle='->', lw=1, color='orange'))
        ax.annotate('', xy=(15, 8.75), xytext=(14, 7.5),
                   arrowprops=dict(arrowstyle='->', lw=1, color='orange'))
        ax.annotate('', xy=(15, 6.75), xytext=(14, 7.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
        
        # 损失组合
        ax.annotate('', xy=(16.25, 3.5), xytext=(16.25, 6),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        ax.annotate('', xy=(16.25, 3.5), xytext=(16.25, 8),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    def _draw_legend(self, ax):
        """绘制图例"""
        legend_elements = [
            patches.Patch(color=self.colors['teacher'], label='教师模型'),
            patches.Patch(color=self.colors['student'], label='学生模型'),
            patches.Patch(color=self.colors['distillation'], label='蒸馏过程'),
            patches.Patch(color=self.colors['data'], label='数据流'),
            patches.Patch(color=self.colors['loss'], label='损失函数'),
            patches.Patch(color=self.colors['output'], label='模型输出')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=10)
    
    def create_loss_function_diagram(self):
        """创建损失函数详细图"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 标题
        ax.text(7, 9.5, '知识蒸馏损失函数详解', 
                fontsize=18, fontweight='bold', ha='center')
        
        # 温度缩放公式
        ax.text(7, 8.5, r'$P_i(T) = \frac{e^{z_i/T}}{\sum_{j=1}^{N} e^{z_j/T}}$', 
                fontsize=16, ha='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
        ax.text(7, 8, '温度缩放Softmax', fontsize=12, ha='center', style='italic')
        
        # KL散度公式
        ax.text(3.5, 6.5, r'$L_{distill} = T^2 \cdot KL(P_{student}(T), P_{teacher}(T))$', 
                fontsize=14, ha='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
        ax.text(3.5, 6, '蒸馏损失 (KL散度)', fontsize=11, ha='center', style='italic')
        
        # 交叉熵公式
        ax.text(10.5, 6.5, r'$L_{hard} = CrossEntropy(P_{student}(T=1), y_{true})$', 
                fontsize=14, ha='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen'))
        ax.text(10.5, 6, '硬目标损失 (交叉熵)', fontsize=11, ha='center', style='italic')
        
        # 总损失公式
        ax.text(7, 4, r'$L_{total} = \alpha \cdot L_{distill} + \beta \cdot L_{hard}$', 
                fontsize=16, ha='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral'))
        ax.text(7, 3.5, '总损失函数', fontsize=12, ha='center', style='italic', fontweight='bold')
        
        # 参数说明
        param_text = (
            "参数说明:\n"
            "• T: 温度参数 (通常2.0-6.0)\n"
            "• α: 蒸馏损失权重 (通常0.7-0.9)\n"
            "• β: 硬目标损失权重 (通常0.1-0.3)\n"
            "• α + β = 1"
        )
        ax.text(7, 2, param_text, fontsize=11, ha='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_architecture_comparison(self):
        """创建架构对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        
        # 教师模型架构
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 15)
        ax1.axis('off')
        ax1.set_title('教师模型架构', fontsize=16, fontweight='bold', color=self.colors['teacher'])
        
        # 绘制教师模型层
        for i in range(12):
            y_pos = 13 - i
            layer_box = FancyBboxPatch(
                (2, y_pos-0.4), 6, 0.8,
                boxstyle="round,pad=0.05",
                facecolor=self.colors['teacher'],
                alpha=0.7,
                edgecolor='black'
            )
            ax1.add_patch(layer_box)
            ax1.text(5, y_pos, f'Transformer Layer {i+1}', 
                    ha='center', va='center', fontsize=9)
        
        # 教师模型参数
        ax1.text(5, 0.5, '参数: 44M\nd_model: 512\n注意力头: 8', 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 学生模型架构
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 15)
        ax2.axis('off')
        ax2.set_title('学生模型架构', fontsize=16, fontweight='bold', color=self.colors['student'])
        
        # 绘制学生模型层
        for i in range(6):
            y_pos = 13 - i*2
            layer_box = FancyBboxPatch(
                (2, y_pos-0.4), 6, 0.8,
                boxstyle="round,pad=0.05",
                facecolor=self.colors['student'],
                alpha=0.7,
                edgecolor='black'
            )
            ax2.add_patch(layer_box)
            ax2.text(5, y_pos, f'Transformer Layer {i+1}', 
                    ha='center', va='center', fontsize=9)
        
        # 学生模型参数
        ax2.text(5, 0.5, '参数: 10M\nd_model: 256\n注意力头: 4', 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig

def main():
    """主函数 - 生成所有可视化图表"""
    visualizer = DistillationVisualizer()
    
    # 创建综合流程图
    fig1 = visualizer.create_comprehensive_diagram()
    fig1.savefig('distillation_comprehensive_flow.png', dpi=300, bbox_inches='tight')
    print("✅ 综合流程图已保存: distillation_comprehensive_flow.png")
    
    # 创建损失函数图
    fig2 = visualizer.create_loss_function_diagram()
    fig2.savefig('distillation_loss_function.png', dpi=300, bbox_inches='tight')
    print("✅ 损失函数图已保存: distillation_loss_function.png")
    
    # 创建架构对比图
    fig3 = visualizer.create_architecture_comparison()
    fig3.savefig('distillation_architecture_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 架构对比图已保存: distillation_architecture_comparison.png")
    
    plt.show()

if __name__ == "__main__":
    main()
