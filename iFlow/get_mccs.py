from os import listdir, path
import argparse
import json

def get_mccs(directory):
    mcc_tups = []
    for exp_dir in listdir(directory):
        logfile = path.join(directory, exp_dir + '/log/log.json')
        try:
            with open(logfile, 'rb') as f:
                jsonfile = json.load(f)
                mcc = jsonfile['perf']
                seed = jsonfile['metadata']['file'].split('_')[6]
                mcc_tups.append( (exp_dir, float(mcc), int(seed)))
        except:
            mcc_tups.append( (exp_dir, -1, -1) )

    mcc_tups = sorted(mcc_tups, key = lambda x: x[1], reverse = True)
    return mcc_tups

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', metavar='dir', type=str,
                    help='directory with all experiments')
    parser.add_argument('-w', '--write', action='store_true', default=False, help='run without logging')
    args = parser.parse_args()

    if args.write:
        with open('mcc_results.txt', 'w') as f:
            mcc_tups = get_mccs(args.dir)
            for exp_dir, mcc, seed in mcc_tups:
                f.write(str(exp_dir) + ' : ' + str(mcc) + ', ' + str(seed) + '\n')
    else:
        mcc_tups = get_mccs(args.dir)
        for exp_dir, mcc, seed in mcc_tups:
            print(exp_dir, ':', mcc, seed)
