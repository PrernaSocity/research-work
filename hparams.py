from text import symbols


class hparams:
    seed = 0

    ################################
    # Data Parameters              #
    ################################
    text_cleaners=['english_cleaners']

    ################################
    # Audio                        #
    ################################
    num_mels = 80
    num_freq = 513
    sample_rate = 22050
    frame_shift = 256
    frame_length = 1024
    fmin = 0
    fmax = 8000
    power = 1.5
    gl_iters = 30

    ################################
    # Train                        #
    ################################
    is_cuda = True
    pin_mem = True
    n_workers = 4
    prep = True
    pth = 'lj-22k.pkl'
    lr = 2e-3
    betas = (0.9, 0.999)
    eps = 1e-6
    sch = True
    sch_step = 4000
    max_iter = 200
    batch_size = 16
    iters_per_log = 10
    iters_per_sample = 500
    iters_per_ckpt = 10000
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    eg_text = 'OMAK is a thinking process which considers things always positively.'

    ################################
    # Model Parameters             #
    ################################
    n_symbols = len(symbols)
    symbols_embedding_dim = 512

    
    encoder_kernel_size = 5
    encoder_n_convolutions = 3
    encoder_embedding_dim = 512

    
    n_frames_per_step = 3
    decoder_rnn_dim = 1024
    prenet_dim = 256
    max_decoder_ratio = 10
    gate_threshold = 0.5
    p_attention_dropout = 0.1
    p_decoder_dropout = 0.1

    
    attention_rnn_dim = 1024
    attention_dim = 128

    
    attention_location_n_filters = 32
    attention_location_kernel_size = 31

    
    postnet_embedding_dim = 512
    postnet_kernel_size = 5
    postnet_n_convolutions = 5

