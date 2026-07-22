import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    '容重 Y(kg/m3)', 
    '粘聚力 C(kPa)', 
    '内摩擦角 φ(°)', 
    '坡角 β(°)', 
    '坡高 H(m)', 
    '孔隙水压力比 r.'
]

ENGLISH_COLUMN_MAP = {
    'unit_weight': '容重 Y(kg/m3)',
    'cohesion': '粘聚力 C(kPa)',
    'internal_friction_angle': '内摩擦角 φ(°)',
    'slope_angle': '坡角 β(°)',
    'slope_height': '坡高 H(m)',
    'pore_water_pressure_ratio': '孔隙水压力比 r.'
}

def create_features(X):
    X_new = X.copy()
    
    X_new['C_phi'] = X['粘聚力 C(kPa)'] * X['内摩擦角 φ(°)']
    X_new['Y_H'] = X['容重 Y(kg/m3)'] * X['坡高 H(m)']
    X_new['beta_H'] = X['坡角 β(°)'] / (X['坡高 H(m)'] + 0.1)
    X_new['C_Y'] = X['粘聚力 C(kPa)'] / (X['容重 Y(kg/m3)'] + 0.1)
    X_new['phi_beta'] = X['内摩擦角 φ(°)'] / (X['坡角 β(°)'] + 0.1)
    X_new['r_C'] = X['孔隙水压力比 r.'] * X['粘聚力 C(kPa)']
    X_new['H_phi'] = X['坡高 H(m)'] / (X['内摩擦角 φ(°)'] + 0.1)
    X_new['Y_beta'] = X['容重 Y(kg/m3)'] * X['坡角 β(°)']
    
    X_new['C_phi_beta'] = X_new['C_phi'] / (X['坡角 β(°)'] + 0.1)
    X_new['Y_H_beta'] = X_new['Y_H'] / (X['坡角 β(°)'] + 0.1)
    X_new['stability_index'] = (X['粘聚力 C(kPa)'] * X['内摩擦角 φ(°)']) / (X['坡高 H(m)'] * X['坡角 β(°)'] + 0.1)
    X_new['factor_H'] = X['坡高 H(m)'] * X['孔隙水压力比 r.']
    X_new['C_r_Y'] = X['粘聚力 C(kPa)'] / (X['容重 Y(kg/m3)'] * (X['孔隙水压力比 r.'] + 0.01) + 0.1)
    
    X_new['tan_phi'] = np.tan(np.radians(X['内摩擦角 φ(°)']))
    X_new['tan_beta'] = np.tan(np.radians(X['坡角 β(°)']))
    X_new['phi_beta_ratio'] = X_new['tan_phi'] / (X_new['tan_beta'] + 0.01)
    
    X_new['C_H_Y'] = X['粘聚力 C(kPa)'] / (X['坡高 H(m)'] * X['容重 Y(kg/m3)'] + 0.1)
    X_new['r_beta'] = X['孔隙水压力比 r.'] * X['坡角 β(°)']
    X_new['r_H'] = X['孔隙水压力比 r.'] * X['坡高 H(m)']
    
    X_new['log_H'] = np.log1p(X['坡高 H(m)'])
    X_new['sqrt_C'] = np.sqrt(X['粘聚力 C(kPa)'])
    X_new['sqrt_phi'] = np.sqrt(X['内摩擦角 φ(°)'])
    
    X_new['C2'] = X['粘聚力 C(kPa)'] ** 2
    X_new['phi2'] = X['内摩擦角 φ(°)'] ** 2
    X_new['H2'] = X['坡高 H(m)'] ** 2
    X_new['beta2'] = X['坡角 β(°)'] ** 2
    
    X_new['C_sqrt_phi'] = X['粘聚力 C(kPa)'] * np.sqrt(X['内摩擦角 φ(°)'])
    X_new['Y_sqrt_H'] = X['容重 Y(kg/m3)'] * np.sqrt(X['坡高 H(m)'])
    
    X_new['sin_beta'] = np.sin(np.radians(X['坡角 β(°)']))
    X_new['cos_beta'] = np.cos(np.radians(X['坡角 β(°)']))
    X_new['sin_phi'] = np.sin(np.radians(X['内摩擦角 φ(°)']))
    X_new['cos_phi'] = np.cos(np.radians(X['内摩擦角 φ(°)']))
    
    X_new['safety_factor_approx'] = (X['粘聚力 C(kPa)'] + X['容重 Y(kg/m3)'] * X['坡高 H(m)'] * np.tan(np.radians(X['内摩擦角 φ(°)']))) / (X['容重 Y(kg/m3)'] * X['坡高 H(m)'] * np.sin(np.radians(X['坡角 β(°)'])) + 0.1)
    
    X_new['C_cubed'] = X['粘聚力 C(kPa)'] ** 1.5
    X_new['phi_cubed'] = X['内摩擦角 φ(°)'] ** 1.5
    X_new['H_cubed'] = X['坡高 H(m)'] ** 1.5
    
    X_new['C_phi_H'] = X_new['C_phi'] / (X['坡高 H(m)'] + 0.1)
    X_new['Y_phi'] = X['容重 Y(kg/m3)'] * X['内摩擦角 φ(°)']
    X_new['C_beta'] = X['粘聚力 C(kPa)'] / (X['坡角 β(°)'] + 0.1)
    
    return X_new


def prepare_input_data(unit_weight, cohesion, internal_friction_angle,
                       slope_angle, slope_height, pore_water_pressure_ratio):
    input_data = {
        '容重 Y(kg/m3)': [unit_weight],
        '粘聚力 C(kPa)': [cohesion],
        '内摩擦角 φ(°)': [internal_friction_angle],
        '坡角 β(°)': [slope_angle],
        '坡高 H(m)': [slope_height],
        '孔隙水压力比 r.': [pore_water_pressure_ratio]
    }
    df = pd.DataFrame(input_data)
    return df