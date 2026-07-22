import argparse
import os
import sys
from ..core.model_utils import EnsemblePredictor
from ..core.feature_utils import create_features, prepare_input_data


def parse_args():
    parser = argparse.ArgumentParser(description='Slope Stability Predictor - 边坡稳定性预测工具')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    predict_parser = subparsers.add_parser('predict', help='预测边坡稳定性')
    predict_parser.add_argument('--model-dir', required=True, help='模型文件目录')
    predict_parser.add_argument('--unit-weight', type=float, required=True, help='容重 γ (kN/m³)')
    predict_parser.add_argument('--cohesion', type=float, required=True, help='粘聚力 C (kPa)')
    predict_parser.add_argument('--friction-angle', type=float, required=True, help='内摩擦角 φ (°)')
    predict_parser.add_argument('--slope-angle', type=float, required=True, help='坡角 β (°)')
    predict_parser.add_argument('--slope-height', type=float, required=True, help='坡高 H (m)')
    predict_parser.add_argument('--pore-pressure', type=float, required=True, help='孔隙水压力比 r_u')
    predict_parser.add_argument('--threshold', type=float, default=0.716, help='预测阈值')
    
    batch_parser = subparsers.add_parser('batch', help='批量预测边坡稳定性')
    batch_parser.add_argument('--model-dir', required=True, help='模型文件目录')
    batch_parser.add_argument('--input', required=True, help='输入数据文件 (Excel)')
    batch_parser.add_argument('--output', default='predictions.xlsx', help='输出结果文件')
    
    check_parser = subparsers.add_parser('check', help='检查模型文件')
    check_parser.add_argument('--model-dir', required=True, help='模型文件目录')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.command == 'predict':
        print('开始预测边坡稳定性...')
        
        try:
            model = EnsemblePredictor(args.model_dir)
            model.load()
            
            input_df = prepare_input_data(
                args.unit_weight,
                args.cohesion,
                args.friction_angle,
                args.slope_angle,
                args.slope_height,
                args.pore_pressure
            )
            
            X_enhanced = create_features(input_df)
            proba = model.predict_proba(X_enhanced)[0]
            is_stable = proba >= args.threshold
            
            print(f'\n预测结果:')
            print(f'  稳定性概率: {proba:.4f} ({proba*100:.2f}%)')
            print(f'  判定阈值: {args.threshold:.3f}')
            print(f'  预测状态: {"✅ 稳定" if is_stable else "❌ 不稳定"}')
            
        except Exception as e:
            print(f'错误: {str(e)}')
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.command == 'batch':
        print('开始批量预测...')
        
        try:
            import pandas as pd
            
            model = EnsemblePredictor(args.model_dir)
            model.load()
            
            df = pd.read_excel(args.input)
            
            required_columns = ['容重 Y(kg/m3)', '粘聚力 C(kPa)', '内摩擦角 φ(°)', 
                               '坡角 β(°)', '坡高 H(m)', '孔隙水压力比 r.']
            
            for col in required_columns:
                if col not in df.columns:
                    print(f'错误: 缺少必需列: {col}')
                    sys.exit(1)
            
            X_enhanced = create_features(df[required_columns])
            proba = model.predict_proba(X_enhanced)
            
            df['稳定性概率'] = proba
            df['预测状态'] = ['稳定' if p >= 0.716 else '不稳定' for p in proba]
            
            df.to_excel(args.output, index=False)
            print(f'预测结果已保存至: {args.output}')
            
        except Exception as e:
            print(f'错误: {str(e)}')
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.command == 'check':
        print('检查模型文件...')
        
        try:
            model = EnsemblePredictor(args.model_dir)
            
            print(f'\n模型目录: {model.model_dir}')
            print(f'目录存在: {"✅ 是" if os.path.exists(model.model_dir) else "❌ 否"}')
            
            if os.path.exists(model.model_dir):
                files = os.listdir(model.model_dir)
                print(f'\n目录内容 ({len(files)}个文件):')
                for f in files:
                    f_path = os.path.join(model.model_dir, f)
                    size = os.path.getsize(f_path)
                    size_str = f"{size / 1024 / 1024:.2f} MB" if size > 1024*1024 else f"{size / 1024:.2f} KB"
                    print(f'  - {f} ({size_str})')
                
                model.load()
                print(f'\n✅ 模型加载成功')
                print(f'   加载的模型: {list(model.models.keys())}')
                print(f'   模型权重: {model.weights}')
                
        except Exception as e:
            print(f'\n❌ 检查失败: {str(e)}')
            sys.exit(1)
    
    else:
        print('请指定命令，使用 --help 查看可用命令')
        sys.exit(1)


if __name__ == '__main__':
    main()