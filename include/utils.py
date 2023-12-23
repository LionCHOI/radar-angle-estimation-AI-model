import torch
import numpy as np

def min_max_cal(Rx_sig, min_percent = 0.025, max_percent =  0.975):
    Rx_sig_cdf =  np.sort(np.abs(Rx_sig.reshape(-1)))
    total_len = len(Rx_sig_cdf)
    cdf_value, cdf_value_idx = np.zeros(total_len), np.zeros(total_len)

    prev_value, cdf_idx, each_cnt = -1, 0, 0
    
    for value in Rx_sig_cdf:
        each_cnt += 1
        if prev_value != value: 
            cdf_value[cdf_idx] = each_cnt/total_len
            cdf_value_idx[cdf_idx] = value
            
            prev_value = value
            cdf_idx += 1
        else:
            cdf_value[cdf_idx] = each_cnt/total_len
            
    max_idx = int(np.where(cdf_value == 1.0)[0])
    cdf_value, cdf_value_idx = cdf_value[:max_idx+1], cdf_value_idx[:max_idx+1]

    min_value, max_value = int(np.max(np.where(cdf_value <= 0.025)[0])), int(np.min(np.where(cdf_value >= 0.975)[0]))   # 2 sigma 

    return cdf_value_idx[min_value], cdf_value_idx[max_value]

def preprocessing_rx_sig(path, min_angle, max_angle, resolution):
    Rx_sig = np.empty(0)
    angle = np.empty(0)

    for angle_value in range(min_angle, max_angle+1, resolution):
        RX_sig_tmp = np.load(path + f'/output_COV_{angle_value}.npy')
        angle_tmp = np.load(path + f'/output_angle_{angle_value}.npy')
        
        Rx_sig_new = np.empty(0)
        Rx_sig_angle = np.empty(0)

        min_cdf_value, max_cdf_value = min_max_cal(RX_sig_tmp)

        for idx, Rx_sig_val in enumerate(RX_sig_tmp):
            RX_sig_flag_min, RX_sig_flag_max = np.abs(Rx_sig_val) >= min_cdf_value, np.abs(Rx_sig_val) <= max_cdf_value
            if np.mean(RX_sig_flag_max.astype(int)) == 1 and np.mean(RX_sig_flag_min) == 1:
                if len(Rx_sig_new) == 0:
                    Rx_sig_new = Rx_sig_val
                    Rx_sig_angle = angle_tmp[idx]
                else:
                    Rx_sig_new = np.concatenate((Rx_sig_new, Rx_sig_val))
                    Rx_sig_angle = np.concatenate((Rx_sig_angle, angle_tmp[idx]))
                    
        Rx_sig_new /= max_cdf_value 
        
        if len(Rx_sig) == 0 :
            Rx_sig = Rx_sig_new
            angle = Rx_sig_angle
        else:
            Rx_sig = np.concatenate((Rx_sig, Rx_sig_new))
            angle = np.concatenate((angle, Rx_sig_angle))
        
    Rx_sig = Rx_sig.reshape(-1, 4, 4)
    
    return Rx_sig, angle

def postprocessing(outputs, num_class, resolution, min_angle):
    ZERO, THRESHOLD_NON_DOMINANT = 0, 0.5
    pred_list = []
    
    values = outputs.topk(num_class)[0]
    idxs = outputs.topk(num_class)[1] * resolution + min_angle

    for val, idx in zip(values, idxs):
        if torch.sum(val) < THRESHOLD_NON_DOMINANT: # 1. dominant한 부분이 없는 경우
            if torch.sum(val) == ZERO:  # 1-1. 합이 0인 경우
                pred_list.append(ZERO)
            else:                       # 1-2. 합이 0이 아닌 경우
                idx_max = torch.max(torch.where(val != 0)[0])
                pred_list.append(np.mean(idx[:idx_max+1].cpu().numpy()))
            
        elif torch.sum(val[1:]) < val[0]:           # 2. dominant한 부분이 있는데 top1이 나머지의 합보다 큰 경우
            pred_list.append(idx[0].item())
            continue
        
        else:                                       # 3. dominant한 부분이 있는데 top1이 나머지의 합보다 작은 경우
            sum_val, sum_idx, idx_list = 0, idx[0].item(), []
            for idx_remainder, val_remainder in enumerate(val[1:], 1):
                sum_val += val_remainder
                sum_idx += idx[idx_remainder].type(torch.float)
                idx_list.append(idx[idx_remainder].cpu())
                
                if val[0] < sum_val:
                    if np.sum(np.array(idx_list) < 7) and np.sum(np.array(idx_list) > 7):
                            pred_list.append(np.round(sum_idx.item()))
                    else:
                        pred_list.append(np.round(sum_idx.item() / (idx_remainder + 1)))
                    break
    return pred_list