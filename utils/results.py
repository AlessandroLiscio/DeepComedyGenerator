import os
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def create_folder(path:str):

    '''create folder if it doesn't exist'''

    if not os.path.exists(path):
        os.mkdir(path)
        print("CREATED: ", path)

def get_model_name(log):

    '''stringify the model description for the file name'''

    return"_".join((
        f"{log['model']['encoders']}",
        f"{log['model']['decoders']}",
        f"{log['model']['heads']}",
        f"{log['model']['d_model']}",
        f"{log['model']['dff']}",
        f"{log['trainings']['production']['repetitions']}",
        f"{log['trainings']['comedy']['repetitions']}"   
    ))

def save_results(log, out_path:str = 'results/'):

    '''save log dictionary as .json file and generations
    texts as .txt files in 'out_path' folder'''

    # stringify the model description for the file name
    model_name = get_model_name(log)

    # create results folder if it doesn't exist
    create_folder(out_path)

    # create generations folder if it doesn't exist
    generations_path = out_path+"generations/"
    create_folder(generations_path)

    # Save the log file 
    log_file = f"{out_path}LOG_{model_name}.json"
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
        file_name = f"GEN-{temperature}_[{model_name}].txt"
        file_path = generations_path + file_name
        with open(file_path, "w+") as out_file:
            out_file.write("\n".join(generated_text[1:]))
            generations_files.append(file_path)
            print(f"generated text at temperature {temperature} saved as {file_name}")
    print(f"\tin folder {generations_path}")

def show_train_info(log):

    '''print training information'''

    print('\nMODEL:')
    for param in log['model']:
        print(f" -- {param}: {log['model'][param]}")
    print('\nTRAINING:')
    for training in log['trainings']:
        print(f" -- {training}")
        for info in log['trainings'][training]:
            if 'history' in info:
                hist = log['trainings'][training][info]
                print(f"   -- {info}: {hist[:3]} ... {hist[-3:]}")
            elif info == 'time':
                t = log['trainings'][training][info]
                print(f"   -- {info}: {int(t/3600)}h {int(t/60%60)}m {int(t%60)}s")
            else:
                print(f"   -- {info}: {log['trainings'][training][info]}")

def tabular_generations(log, out_path='results/'):

    '''print generations in tabular form'''

    # Generations
    generations = []
    temperatures = []
    for temp in log['generations']:
        canto = log['generations'][temp]
        canto = canto.replace(' ,',',')
        canto = canto.replace(' .','.')
        canto = canto.replace(' !','!')
        canto = canto.replace(' ?','?')
        canto = canto.replace(' :',':')
        canto = canto.replace(' ;',';')
        canto = canto.split('\n')
        generations.append(canto)
        temperatures.append(temp)

    # header of the table
    head_line = "\n\t    "
    for temp in temperatures:
        head_line += "{:<45}".format(temp)
    head_line += "\n\n"
    
    # organize by columns
    rows = []
    for row_idx in range(len(generations[0])):
        row = ""
        for temp in range(len(temperatures)):
            row += "{:<45}".format(generations[temp][row_idx])
        rows.append(row)

    # print out
    print(head_line)
    for row in rows:
        print(row)

    # save table to file
    with open(out_path+"gen_table.txt", "w+") as f:
        f.write(head_line)
        for row in rows:
            f.write(row)

def plot_hist(log, out_path='results/'):

    '''plot loss and accuracy histories abd save figure
    in 'out_path' folder as 'history.png' file'''

    # create out folder if it does not exist
    create_folder(out_path)

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

    # save plot as .png and show it
    plt.savefig(out_path+"history.png")
    plt.show()