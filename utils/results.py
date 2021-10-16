import os
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def save_results(log, out_path):

    # stringify the model description for the file name
    model_description = "_".join(
        log["model"]["encoders"],
        log["model"]["decoders"],
        log["model"]["d_model"]
        log["model"]["dff"],
        log["model"]["heads"],
        log["trainings"]["production"]["repetitions"],
        log["trainings"]["comedy"]["repetitions"]   
    )

    # create destination folder if it doesn't exist
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        print("CREATED: ", out_path)

    # Save the log file 
    log_file = f"{out_path}/LOG_{model_description}.json"
    with open(log_file, 'w+') as fp:
        json.dump(log, fp, indent=4)
        print(f"log saved as {log_file}")

    # extract the texts from the log
    generations = []
    for temp in log['generations']:
        canto = log['generations'][temp] 
        generations.append(canto.replace(' ,',',').replace(' .','.').replace(' !','!').replace(' ?','?').replace(' :',':').replace(' ;',';').split('\n'))

    # Save the generations as text files
    generations_files = []
    for temperature, generated_text in zip(log["generations"], generations):
        out_file_name = f"GEN-{temperature}_[{model_description}].txt"
        file_path = f"{out_path}/{out_file_name}"
        with open(file_path, "w+") as out_file:
            out_file.write("\n".join(generated_text[1:]))
            generations_files.append(file_path)
            print(f"generated text at temperature {temperature} saved as {out_file_name}")
    print(f"\tin folder {out_path}")

def show_train_info(log):

    '''print training information'''

    # print model and training information
    print('MODEL:')
    for param in log['model']:
        print(f" -- {param}: {log['model'][param]}")
        print('\nTRAINING:')
    for training in log['trainings']:
        print(f" -- {training}")
        for info in log['trainings'][training]:
            if 'history' in info:
                print(f"   -- {info}: {log['trainings'][training][info][:3]} ... {log['trainings'][training][info][-3:]}")
            elif info == 'time':
                print(f"   -- {info}: {int(log['trainings'][training][info]/3600)}h {int(log['trainings'][training][info]/60%60)}m {int(log['trainings'][training][info]%60)}s")
            else:
                print(f"   -- {info}: {log['trainings'][training][info]}")

def show_generations(log, temperatures):

    '''print generations in tabular form'''

    # extract the texts from the log
    generations = []
    for temp in log['generations']:
        canto = log['generations'][temp] 
        generations.append(canto.replace(' ,',',').replace(' .','.').replace(' !','!').replace(' ?','?').replace(' :',':').replace(' ;',';').split('\n'))

    # header of the table
    head_line = "\t    "
    for temp in temperatures:
        head_line += "{:<45}".format(temp)
        print(head_line+"\n\n")

    # organize by columns
    for row_idx in range(len(generations[0])):
        row = ""
    for temp_idx in range(len(temperatures)):
        row += "{:<45}".format(generations[temp_idx][row_idx])
        print(row)

def plots_hist(loss_history, acc_history):

    '''plot loss and accuracy histories'''

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))

    # loss history
    loss_history = log['trainings']['comedy']['loss_history']
    for i, loss in enumerate(loss_history):
        loss_history[i] = float(loss_history[i])

    # accuracy history
    acc_history = log['trainings']['comedy']['acc_history']
    for i, loss in enumerate(acc_history):
        acc_history[i] = float(acc_history[i]) 

    # plot loss history
    ax0.set_title('Loss History', color='lightblue', fontsize=15, fontweight= 'bold')
    ax0.set_xticks(range(0,len(loss_history),5))
    ax0.grid()
    ax0.plot(loss_history, color='blue')

    # plot accuracy history
    ax1.set_title('Accuracy History', color='orange', fontsize=15, fontweight= 'bold')
    ax1.set_xticks(range(0,len(acc_history),5))
    ax1.set_ylim(top=1)
    ax1.grid()
    ax1.plot(acc_history, color='red')
    plt.show()
