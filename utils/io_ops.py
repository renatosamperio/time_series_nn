
def id_name_from_conf(conf):
    model = conf["model_type"]
    epoch = conf["epoch"]
    percentage = conf["percentage"]
    hidden_size = conf["hidden_size"]

    return id_name(model, epoch, percentage, hidden_size)

def id_name(model, epoch, percentage, hidden_size):
    id_name = model.lower() + \
    "_" + str(epoch) + \
    "_" + str(int(percentage*100)) + \
    "_" + str(hidden_size)

    return id_name