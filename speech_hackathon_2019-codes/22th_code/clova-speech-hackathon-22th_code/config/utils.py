import json

def read_cfg(cfg_file):
    with open(cfg_file, 'r') as f:
        cfg = json.loads(f.read())
        cfg = update_cfg(cfg)
    return cfg

def update_cfg(cfg):
    # this function should be extended everytime we increase the config version
    if cfg["config_version"] == 0:
        cfg = makeVer1(cfg)
    if cfg["config_version"] == 1:
        cfg = makeVer2(cfg)
    if cfg["config_version"] == 2:
        cfg = makeVer3(cfg)
    if cfg["config_version"] == 3:
        return cfg
    else:
        raise NotImplementedError # handle newer version

def makeVer1(old):
    cfg = {}
    cfg["config_version"] = 1    
    #<MODEL>
    model = {}
    model["hidden_size"] = old["hidden_size"]
    model["layer_size"] = old["layer_size"]
    model["dropout"] = old["dropout"]
    model["bidirectional"] = old["bidirectional"]
    model["use_attention"] = old["use_attention"]
    model["max_len"] = old["max_len"]
    cfg["model"] = model
    #</MODEL>
    cfg["batch_size"] = old["batch_size"]
    cfg["workers"] = old["workers"]
    cfg["max_epochs"] = old["max_epochs"]
    cfg["lr"] = old["lr"]
    cfg["teacher_forcing"] = old["teacher_forcing"]
    return cfg

def makeVer2(old):
    cfg = {}
    cfg["config_version"] = 2
    #<MODEL>
    model = {}
    old_model = old["model"]
        #<ENC>
    enc = {}
    enc["layer_size"] = old_model["layer_size"]
    model["enc"] = enc
        #</ENC>
        #<DEC>
    dec = {}
    dec["layer_size"] = old_model["layer_size"]
    dec["use_attention"] = old_model["use_attention"]
    dec["max_len"] = old_model["max_len"]
    model["dec"] = dec
        #</DEC>
    model["rnn_cell"] = "gru"
    model["hidden_size"] = old_model["hidden_size"]
    model["dropout"] = old_model["dropout"]
    model["bidirectional"] = old_model["bidirectional"]
    cfg["model"] = model
    #</MODEL>
    #<DATA>
    data = {}
    data["use_mel_scale"] = False
        #<SPEC_AUGMENT>
    spec_augment = {}
    spec_augment["use"] = False
    data["spec_augment"] = spec_augment
        #</SPEC_AUGMENT>
    cfg["data"] = data
    #</DATA>
    cfg["batch_size"] = old["batch_size"]
    cfg["workers"] = old["workers"]
    cfg["max_epochs"] = old["max_epochs"]
    cfg["lr"] = old["lr"]
    cfg["teacher_forcing"] = old["teacher_forcing"]
    return cfg

def makeVer3(old):
    cfg = old # incremental change
    cfg["config_version"] = 3
    trim = {}
    trim["use"] = False
    cfg["data"]["trim_silence"] = trim
    return cfg
