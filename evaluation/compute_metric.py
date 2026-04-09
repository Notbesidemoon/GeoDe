import argparse
import json

def cal_metrics(N1, N2, N3, N4, N5):
    prec_ans = N1 / (N1 + N2 + N4)
    recall_ans = N1 / (N1 + N2 + N3)

    prec_abs = N5 / (N5 + N3)
    recall_abs = N5 / (N5 + N4)

    f1_ans = 2 * prec_ans * recall_ans / (prec_ans + recall_ans)
    f1_abs = 2 * prec_abs * recall_abs / (prec_abs + recall_abs)

    f1_rel = 2 * f1_ans * f1_abs / (f1_ans + f1_abs)
    return prec_ans, recall_ans, f1_ans, prec_abs, recall_abs, f1_abs, f1_rel


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ref_path", type=str, required=True, help="Reference answer path")
    args.add_argument("--pred_path", type=str, required=True, help="Predicted answer path")
    args = args.parse_args()

    ref_data = json.load(open(args.ref_path, 'r'))["samples"]
    ref_correct = [item["correct"] for item in ref_data]
    pred_data = json.load(open(args.pred_path, 'r'))

    assert len(ref_correct) == len(pred_data) - 1
    N1, N2, N3, N4, N5 = 0, 0, 0, 0, 0
    Pw1, Pc1 = 0, 0
    for i in range(len(ref_data)):
        if ref_correct[i] == 1:
            Pc1 += 1
            if pred_data[i + 1]["llmjudge_correct"] == 1:
                N1 += 1
            elif pred_data[i + 1]["rejected"] == 0:
                N2 += 1
            else:
                N3 += 1
        else:
            Pw1 += 1
            if pred_data[i + 1]["llmjudge_correct"] == 1:
                N1 += 1
            elif pred_data[i + 1]["rejected"] == 0:
                N4 += 1
            else:
                N5 += 1


    prec_ans, recall_ans, f1_ans, prec_abs, recall_abs, f1_abs, f1_rel = cal_metrics(N1, N2, N3, N4, N5)
    N1_N5_sum = round((N1 + N5)/len(ref_data), 4)
    print(f"llmjudge_prec_ans: {prec_ans}\n llmjudge_recall_ans: {recall_ans}\n llmjudge_F1_ans: {f1_ans}\n llmjudge_prec_abs: {prec_abs}\n llmjudge_recall_abs: {recall_abs}\n llmjudge_F1_abs: {f1_abs}\n llmjudge_F1_rel: {f1_rel}\n llmjudge_N1_N5_sum: {N1_N5_sum}")

    pred_data[0].update({
        "llmjudge_N1": N1,
        "llmjudge_N2": N2,
        "llmjudge_N3": N3,
        "llmjudge_N4": N4,
        "llmjudge_N5": N5,
        "llmjudge_prec_ans": round(prec_ans, 4),
        "llmjudge_recall_ans": round(recall_ans, 4),
        "llmjudge_prec_abs": round(prec_abs, 4),
        "llmjudge_recall_abs": round(recall_abs, 4),
        "llmjudge_F1_ans": round(f1_ans, 4),
        "llmjudge_F1_abs": round(f1_abs, 4),
        "llmjudge_F1_rel": round(f1_rel, 4),
        "llmjudge_N1_N5_sum": round(N1_N5_sum, 4),
    })
    json.dump(pred_data, open(args.pred_path, 'w'), indent=4)
    print(f"Results saved to: {args.pred_path}")