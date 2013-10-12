"""
This module provides a simple interface for sanely launching array jobs
"""
import argparse
import errno
import json
import logging
import numpy as np
import os
import subprocess
from itertools import groupby, count


ROOT = os.getenv('DEV_ROOT') + '/fmo-zofe-spectra'

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s:%(levelname)s:%(message)s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task-id", type=int, default=0)
    parser.add_argument("-j", "--job-id", type=int, default=0)
    parser.add_argument("-N", "--job-name", type=str, default='UNNAMED')
    parser.add_argument('--override', type=json.loads)
    parser.add_argument("--print-params", action='store_true')
    parser.add_argument("--num-tasks", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument('--matching-tasks', type=json.loads, default='{}')
    args = parser.parse_args()
    args.skip_eval = args.print_params or args.num_tasks or args.matching_tasks
    args.debug = args.debug
    return args


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_json(filename, contents):
    with open(filename, 'w') as f:
        json.dump(contents, f, sort_keys=True, indent=4,
                  cls=NumpyAwareJSONEncoder)


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def sequential_ranges(ids):
    c = count()
    key_func = lambda n: n - next(c)
    for _, g in groupby(sorted(ids), key_func):
        g = list(g)
        if len(g) > 1:
            yield '{0}-{1}'.format(g[0], g[-1])
        else:
            yield '{0}'.format(g[0])


def matching_parameter_ids(match, ids, parameters):
    for i, params in zip(ids, parameters):
        if nested_items_match(match, params):
            yield i


def nested_items_match(match, params):
    for k, v in match.iteritems():
        if k not in params:
            return False
        elif isinstance(v, dict):
            if not nested_items_match(v, params[k]):
                return False
        elif v != params[k]:
            return False
    return True


def run_task(main, all_parameters):
    args = parse_args()

    if args.skip_eval:
        if args.print_params:
            print json.dumps(all_parameters, sort_keys=True, indent=4)
        if args.num_tasks:
            print len(all_parameters)
        if args.matching_tasks:
            ids = matching_parameter_ids(args.matching_tasks,
                                         range(1, len(all_parameters) + 1),
                                         all_parameters)
            print ','.join(sequential_ranges(ids))
        return

    logging.info('parsed args: {0}'.format(vars(args)))

    job_root = '{root}/results/{job_name}/{job_id}'.format(
        root=ROOT, **vars(args))
    make_sure_path_exists(job_root)

    task_root = '{job_root}/task_{task_id:03d}'.format(
        job_root=job_root, **vars(args))
    logging.info('results will be saved to ' + task_root)

    try:
        params = all_parameters[args.task_id - 1]
    except IndexError:
        logging.error('job_id outside of expected range')
        raise

    if args.override:
        logging.info('overriding params with {0}'.format(args.override))
        params = dict(params, **args.override)

    commit_str = subprocess.check_output(
        ['git', '--git-dir={}/.git'.format(ROOT),
         'log', '-n', '1', '--pretty=oneline']).strip()
    logging.info('git commit: {}'.format(commit_str))

    logging.info('current params: {}'.format(params))
    save_json(task_root + '.json', params)

    try:
        params['results'] = main(task_root=task_root, debug=args.debug,
                                 **params)
        params['commit'] = commit_str
    except:
        logging.exception('task failed')
        if args.debug:
            raise
    else:
        logging.info('results: {0}'.format(params))
        save_json(task_root + '.json', params)
        logging.info('results saved')
