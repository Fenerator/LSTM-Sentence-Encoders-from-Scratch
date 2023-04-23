output = {
    "MR": {"devacc": 56.95, "acc": 56.57, "ndev": 10662, "ntest": 10662},
    "CR": {"devacc": 65.54, "acc": 64.9, "ndev": 3775, "ntest": 3775},
    "SUBJ": {"devacc": 70.33, "acc": 70.76, "ndev": 10000, "ntest": 10000},
    "MPQA": {"devacc": 77.65, "acc": 77.47, "ndev": 10606, "ntest": 10606},
    "SST2": {"devacc": 57.0, "acc": 58.21, "ndev": 872, "ntest": 1821},
    "TREC": {"devacc": 22.93, "acc": 18.8, "ndev": 5452, "ntest": 500},
    "SICKEntailment": {"devacc": 56.4, "acc": 56.69, "ndev": 500, "ntest": 4927},
    "MRPC": {"devacc": 67.54, "acc": 66.49, "f1": 79.87, "ndev": 4076, "ntest": 1725},
}


keys = ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "SICKEntailment", "MRPC"]

for key in keys:
    devacc = output[key]["devacc"] / 100
    acc = output[key]["acc"] / 100
    print(f"{devacc}")
    print(f"{acc}")
