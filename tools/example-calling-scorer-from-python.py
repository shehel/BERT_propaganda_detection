import task3_scorer_onefile

class Args:
    submission = "example-submission-task3-predictions-test.txt"
    gold = "data/test-task3-labels"
    log_file = None
    debug_on_std = None

args = Args()
f1=task3_scorer_onefile.main(args)
print(f1)
