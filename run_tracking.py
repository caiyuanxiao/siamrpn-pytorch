from __future__ import absolute_import

from got10k.experiments import *

from siamrpn import TrackerSiamRPN


if __name__ == '__main__':    #name为当前模块名，当模块被直接运行时，以下代码被执行，当模块被导入时，以下代码不运行
    # setup tracker
    net_path = 'pretrained/siamrpn/model.pth'
    tracker = TrackerSiamRPN(net_path=net_path)

    # setup experiments
    # 7 datasets with different versions
    experiments = [
        ExperimentGOT10k('data/GOT-10k', subset='test'),
        ExperimentOTB('data/OTB', version=2015),
        ExperimentOTB('data/OTB', version=2013),
        ExperimentVOT('data/vot2018', version=2018),
        ExperimentUAV123('data/UAV123', version='UAV123'),
        ExperimentUAV123('data/UAV123', version='UAV20L'),
        ExperimentDTB70('data/DTB70'),
        ExperimentTColor128('data/Temple-color-128'),
        ExperimentNfS('data/nfs', fps=30),
        ExperimentNfS('data/nfs', fps=240),
    ]

    # run experiments
    for e in experiments:
        e.run(tracker, visualize=True)
        e.report([tracker.name])
