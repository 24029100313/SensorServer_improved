#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def visualize_trajectory(trajectory_file, title=None, show=True, save_path=None):
    """可视化轨迹数据"""
    # 加载轨迹数据
    with open(trajectory_file, 'r') as f:
        trajectory_data = json.load(f)
    
    # 检查轨迹数据是否为空
    if not trajectory_data:
        print(f"警告: {trajectory_file} 中没有轨迹数据")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.text(0, 0, 0, "没有轨迹数据", fontsize=14, ha='center')
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('相机轨迹 (无数据)', fontsize=14)
            
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
            
        return fig, ax
    
    # 提取位置数据
    positions = np.array([data['position'] for data in trajectory_data])
    
    # 创建3D图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='轨迹')
    
    # 标记起点和终点
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', marker='o', s=100, label='起点')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', marker='o', s=100, label='终点')
    
    # 每隔一定数量的点标记一个位置点
    step = max(1, len(positions) // 20)  # 最多标记20个点
    for i in range(0, len(positions), step):
        ax.text(positions[i, 0], positions[i, 1], positions[i, 2], f'{i}', fontsize=8)
    
    # 设置图表属性
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('相机轨迹', fontsize=14)
    
    ax.legend(fontsize=12)
    
    # 设置坐标轴比例相等
    max_range = np.max([
        np.max(positions[:, 0]) - np.min(positions[:, 0]),
        np.max(positions[:, 1]) - np.min(positions[:, 1]),
        np.max(positions[:, 2]) - np.min(positions[:, 2])
    ])
    
    mid_x = (np.max(positions[:, 0]) + np.min(positions[:, 0])) / 2
    mid_y = (np.max(positions[:, 1]) + np.min(positions[:, 1])) / 2
    mid_z = (np.max(positions[:, 2]) + np.min(positions[:, 2])) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图表
    if show:
        plt.show()
    
    return fig, ax

def visualize_all_movements(session_dir, save_dir=None, show=True):
    """可视化会话中所有运动类型的轨迹"""
    # 获取所有运动类型
    movement_types = [d for d in os.listdir(session_dir) 
                     if os.path.isdir(os.path.join(session_dir, d)) and not d.startswith(".")]
    
    print(f"找到以下运动类型: {', '.join(movement_types)}")
    
    # 为每种运动类型创建一个子图
    fig = plt.figure(figsize=(18, 12))
    
    for i, movement in enumerate(movement_types, 1):
        # 轨迹文件路径
        trajectory_file = os.path.join(session_dir, movement, "vio_results", "trajectory.json")
        
        if not os.path.exists(trajectory_file):
            print(f"警告: 找不到 {movement} 的轨迹文件")
            continue
        
        # 加载轨迹数据
        with open(trajectory_file, 'r') as f:
            trajectory_data = json.load(f)
        
        # 提取位置数据
        positions = np.array([data['position'] for data in trajectory_data])
        
        # 创建子图
        ax = fig.add_subplot(2, 3, i, projection='3d')
        
        # 绘制轨迹
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
        
        # 标记起点和终点
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', marker='o', s=50, label='起点')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', marker='o', s=50, label='终点')
        
        # 设置图表属性
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(movement)
        
        if i == 1:  # 只在第一个子图显示图例
            ax.legend()
    
    plt.suptitle('所有运动类型的相机轨迹', fontsize=16)
    plt.tight_layout()
    
    # 保存图表
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "all_trajectories.png"), dpi=300, bbox_inches='tight')
    
    # 显示图表
    if show:
        plt.show()
    
    return fig

def compare_movements(session_dir, movement_types=None, save_dir=None, show=True):
    """比较不同运动类型的轨迹"""
    # 如果没有指定运动类型，则使用所有可用的运动类型
    if movement_types is None:
        movement_types = [d for d in os.listdir(session_dir) 
                         if os.path.isdir(os.path.join(session_dir, d)) and not d.startswith(".")]
    
    # 创建3D图
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 颜色映射
    colors = plt.cm.jet(np.linspace(0, 1, len(movement_types)))
    
    for i, movement in enumerate(movement_types):
        # 轨迹文件路径
        trajectory_file = os.path.join(session_dir, movement, "vio_results", "trajectory.json")
        
        if not os.path.exists(trajectory_file):
            print(f"警告: 找不到 {movement} 的轨迹文件")
            continue
        
        # 加载轨迹数据
        with open(trajectory_file, 'r') as f:
            trajectory_data = json.load(f)
        
        # 提取位置数据
        positions = np.array([data['position'] for data in trajectory_data])
        
        # 绘制轨迹
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-', 
                color=colors[i], linewidth=2, label=movement)
        
        # 标记起点和终点
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  color=colors[i], marker='o', s=50)
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  color=colors[i], marker='s', s=50)
    
    # 设置图表属性
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('不同运动类型的轨迹比较', fontsize=14)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    # 保存图表
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "trajectory_comparison.png"), dpi=300, bbox_inches='tight')
    
    # 显示图表
    if show:
        plt.show()
    
    return fig, ax

def main():
    parser = argparse.ArgumentParser(description='可视化VIO结果')
    parser.add_argument('--session_dir', type=str, help='数据会话目录路径')
    parser.add_argument('--movement', type=str, help='要可视化的特定运动类型')
    parser.add_argument('--compare', action='store_true', help='比较所有运动类型的轨迹')
    parser.add_argument('--save_dir', type=str, help='保存可视化结果的目录')
    parser.add_argument('--no_show', action='store_true', help='不显示图表（仅保存）')
    
    args = parser.parse_args()
    
    # 如果没有指定会话目录，则使用最新的会话
    if args.session_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collected_data")
        session_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("session_")]
        if not session_dirs:
            print(f"错误: 在 {base_dir} 中找不到会话目录")
            return
        
        session_dir = max(session_dirs, key=os.path.getctime)
        print(f"使用最新的数据会话: {session_dir}")
    else:
        session_dir = args.session_dir
    
    # 确保会话目录存在
    if not os.path.exists(session_dir):
        print(f"错误: 会话目录 {session_dir} 不存在")
        return
    
    # 设置保存目录
    save_dir = args.save_dir if args.save_dir else os.path.join(session_dir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置是否显示图表
    show = not args.no_show
    
    if args.movement:
        # 可视化特定运动类型
        trajectory_file = os.path.join(session_dir, args.movement, "vio_results", "trajectory.json")
        if not os.path.exists(trajectory_file):
            print(f"错误: 找不到 {args.movement} 的轨迹文件")
            return
        
        visualize_trajectory(
            trajectory_file, 
            title=f"{args.movement} 轨迹", 
            show=show, 
            save_path=os.path.join(save_dir, f"{args.movement}_trajectory.png")
        )
    elif args.compare:
        # 比较所有运动类型
        compare_movements(session_dir, save_dir=save_dir, show=show)
    else:
        # 可视化所有运动类型
        visualize_all_movements(session_dir, save_dir=save_dir, show=show)
    
    print(f"可视化结果已保存到: {save_dir}")

if __name__ == "__main__":
    main()
