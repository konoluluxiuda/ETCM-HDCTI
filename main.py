from util.config import ModelConf
from util.gpu import configure_cuda_environment
from util.reproducibility import seed_python_numpy


if __name__ == '__main__':


    import time
    s = time.time()
    #Register your model here and add the conf file yuinto the config directory

    try:
        conf = ModelConf('HDCTI.conf')
    except KeyError:
        print('wrong num!')
        exit(-1)
    seed_python_numpy(int(conf['random.seed']) if conf.contains('random.seed') else 2026)
    configure_cuda_environment(conf)

    from HDR import HDR

    recSys = HDR(conf)
    recSys.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
