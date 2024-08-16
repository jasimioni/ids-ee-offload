
import pandas as pd

def get_model_results(df, normal_threshold_exit1, attack_threshold_exit1, normal_threshold_exit2, attack_threshold_exit2):
    # Return the results of a model for specific thresholds. Should pass a df with the data
    f_n_exit1 = 'y_exit_1 == 0 and cnf_exit_1 >= @normal_threshold_exit1'
    f_a_exit1 = 'y_exit_1 == 1 and cnf_exit_1 >= @attack_threshold_exit1'

    f_n_exit2 = 'y_exit_2 == 0 and cnf_exit_2 >= @normal_threshold_exit2'
    f_a_exit2 = 'y_exit_2 == 1 and cnf_exit_2 >= @attack_threshold_exit2'

    exit1_normal = df.query(f_n_exit1)
    exit1_attack = df.query(f_a_exit1)

    exit2 = df.query(f'not ({f_n_exit1}) and not ({f_a_exit1})')

    exit2_normal = exit2.query(f_n_exit2)
    exit2_attack = exit2.query(f_a_exit2)

    not_accepted = exit2.query(f'not ({f_n_exit2}) and not ({f_a_exit2})')

    total = df['y'].count()

    exit1_normal_cnt = exit1_normal['y'].count()
    exit1_attack_cnt = exit1_attack['y'].count()
    exit2_normal_cnt = exit2_normal['y'].count()
    exit2_attack_cnt = exit2_attack['y'].count()

    exit1_rate = ( exit1_normal_cnt + exit1_attack_cnt ) / total

    accepted = exit1_normal_cnt + exit1_attack_cnt + exit2_normal_cnt + exit2_attack_cnt

    acceptance_rate = accepted / total

    fp = exit1_attack.query('y == 0')['y'].count() + exit2_attack.query('y == 0')['y'].count()
    fn = exit1_normal.query('y == 1')['y'].count() + exit2_normal.query('y == 1')['y'].count()

    tn = exit1_normal.query('y == 0')['y'].count() + exit2_normal.query('y == 0')['y'].count()

    tp = exit1_attack.query('y == 1')['y'].count() + exit2_attack.query('y == 1')['y'].count()

    correct = tn + tp

    print(f"Correct: {correct}, FP: {fp}, FN: {fn}, TN: {tn}, TP: {tp}, TP + TN: {tp + tn}")
    
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * precision * recall / (precision + recall)

    accuracy = correct / accepted # Should we consider error from all?

    '''
    exit1_total_time = exit1_normal['bb_time_exit_1'].sum() + exit1_normal['exit_time_exit_1'].sum() + \
                       exit1_attack['bb_time_exit_1'].sum() + exit1_attack['exit_time_exit_1'].sum()

    exit2_total_time = exit2_normal['bb_time_exit_1'].sum() + exit2_normal['bb_time_exit_2'].sum() + exit2_normal['exit_time_exit_2'].sum() + \
                       exit2_attack['bb_time_exit_1'].sum() + exit2_attack['bb_time_exit_2'].sum() + exit2_attack['exit_time_exit_2'].sum()

    not_accepted_total_time = not_accepted['bb_time_exit_1'].sum() + not_accepted['bb_time_exit_2'].sum() + not_accepted['exit_time_exit_2'].sum()

    total_time = exit1_total_time + exit2_total_time + not_accepted_total_time
    '''

    # print(f"Total: {total}")
    # print(f"exit1_normal_cnt: {exit1_normal_cnt}, exit1_attack_cnt: {exit1_attack_cnt}")
    # print(f"exit2_normal_cnt: {exit2_normal_cnt}, exit2_attack_cnt: {exit2_attack_cnt}")
    # print(f"Accepted: {accepted}, Accepted: {total - not_accepted['y'].count()}")
    # print(f"exit1_total_time: {exit1_total_time:.4f}, exit2_total_time: {exit2_total_time:.4f}, not_accepted_total_time: {not_accepted_total_time:.4f}")
    # print(f"exit1_rate: {100 * ( exit1_normal_cnt + exit1_attack_cnt ) / total:.2f}, exit2_rate: {100 * ( exit2_normal_cnt + exit2_attack_cnt ) / total:.2f}")
    # print(f"Accuracy: {100 * accuracy:.2f}, Acceptance: {100 * acceptance_rate:.2f}, Average Time: {1e6 * total_time / total:.2f}")

    return [ accuracy, fpr, fnr, precision, recall, f1, acceptance_rate, exit1_rate, tpr, tnr ]

