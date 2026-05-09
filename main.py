from src.solver.trainer import train_neumatc, test_model, baseline_test

if __name__ == '__main__':
    # 超参数
    MATRIX_SIZE = 256
    OP_TYPE = 'inv'  # 'inv' 或 'svd'
    EQUATION_TYPE = 'inv'  # 'AX=B', 'XA=B', 'AXB=C' 或 'inv'
    
    # 训练
    print(f"开始训练 NeuMatC (方程类型: {EQUATION_TYPE})...")
    model, train_info = train_neumatc(n=MATRIX_SIZE, op=OP_TYPE, equation_type=EQUATION_TYPE)
    
    # 测试
    print("\nNeuMatC 测试:")
    test_model(model, n=MATRIX_SIZE, op=OP_TYPE, equation_type=EQUATION_TYPE)
    
    print("\nNumPy基线测试:")
    baseline_test(n=MATRIX_SIZE, op=OP_TYPE)
