import logging

from videoanalyst.engine.tester.tester_base import TESTERS

logger = logging.getLogger('global')


def test(parsed_args, common_cfg, exp_cfg):
    tester_name = parsed_args.dataset
    logger.info("Start %s" % tester_name)
    TESTERS[tester_name](parsed_args, common_cfg, exp_cfg)
