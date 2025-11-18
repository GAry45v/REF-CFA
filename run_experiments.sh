#!/bin/bash

# 确保在一个实验失败时立即停止
set -e

# --- 在这里配置您想运行的模式 ---
#
# 运行 "分类" (Transformer 训练 + 评估)
MODE_ARGS="--detection"
#
# 运行 "定位" (生成热力图)
# MODE_ARGS="--localize --use_final_residual"
#
# ---------------------------------

# --- VisA 实验列表 ---
# echo "--- 🚀 正在启动实验: capsules, 1000 ---"
# python main_transformer.py $MODE_ARGS data.category=capsules model.load_chp=1000

# echo "--- 🚀 正在启动实验: cashew, 1750 ---"
# python main_transformer.py $MODE_ARGS data.category=cashew model.load_chp=1750

# echo "--- 🚀 正在启动实验: chewinggum, 1250 ---"
# (注意: 假设您的类别名叫 'chewinggum', 而不是 'Chewing gum')
# python main_transformer.py $MODE_ARGS data.category=chewinggum model.load_chp=1250

# echo "--- 🚀 正在启动实验: fryum, 1000 ---"
# python main_transformer.py $MODE_ARGS data.category=fryum model.load_chp=1000

# echo "--- 🚀 正在启动实验: macaroni1, 500 ---"
# python main_transformer.py $MODE_ARGS data.category=macaroni1 model.load_chp=500

# echo "--- 🚀 正在启动实验: macaroni2, 500 ---"
# python main_transformer.py $MODE_ARGS data.category=macaroni2 model.load_chp=500

# echo "--- 🚀 正在启动实验: pcb1, 500 ---"
# python main_transformer.py $MODE_ARGS data.category=pcb1 model.load_chp=500

# echo "--- 🚀 正在启动实验: pcb2, 500 ---"
# python main_transformer.py $MODE_ARGS data.category=pcb2 model.load_chp=500

# echo "--- 🚀 正在启动实验: pcb3, 500 ---"
# python main_transformer.py $MODE_ARGS data.category=pcb3 model.load_chp=500

# echo "--- 🚀 正在启动实验: pcb4, 500 ---"
# python main_transformer.py $MODE_ARGS data.category=pcb4 model.load_chp=500

# echo "--- 🚀 正在启动实验: pipe_fryum, 500 ---"
# (注意: 假设您的类别名叫 'pipe_fryum', 而不是 'Pipe fryum')
# python main_transformer.py $MODE_ARGS data.category=pipe_fryum model.load_chp=500

# --- 🏁 所有实验已完成 ---
# echo "--- 🏁 所有 VisA 实验已完成 ---"
# echo "--- 开始训练Unet ---"
echo "--- 开始在Class1上训练Unet ---"
python main_transformer.py $MODE_ARGS data.category=Class1

echo "--- 开始在Class2上训练Unet ---"
python main_transformer.py $MODE_ARGS data.category=Class2

echo "--- 开始在Class3上训练Unet ---"
python main_transformer.py $MODE_ARGS data.category=Class3

echo "--- 开始在Class4上训练Unet ---"
python main_transformer.py $MODE_ARGS data.category=Class4

echo "--- 开始在Class5上训练Unet ---"
python main_transformer.py $MODE_ARGS data.category=Class5

echo "--- 开始在Class6上训练Unet ---"
python main_transformer.py $MODE_ARGS data.category=Class6