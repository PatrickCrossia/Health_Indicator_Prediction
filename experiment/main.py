from experiment.predictor_baseline import *
from experiment.predictor_advance import *
from experiment.utils import *
import os, time


def revision(Repo, Directory, metrics, repeats, goal, month, tocsv):
    data = data_goal_arrange(Repo, Directory, goal)

    for way in range(10):

        if way == 0:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(LNR(data, month)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            if tocsv:
                with open("../result_revision/mon{}_g{}_mt{}_{}.csv".format(month, goal, metrics, repo[:-12]),
                          "a+") as output:
                    output.write("LNR" + "\n")
                    for i in range(len(list_output)):
                        if i < len(list_output) - 1:
                            output.write(str(list_output[i]) + " ")
                        else:
                            output.write(str(list_output[i]) + "\n\n")
            print("LNR", list_output)

        if way == 1:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(SVM(data, month)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            if tocsv:
                with open("../result_revision/mon{}_g{}_mt{}_{}.csv".format(month, goal, metrics, repo[:-12]),
                          "a+") as output:
                    output.write("SVR" + "\n")
                    for i in range(len(list_output)):
                        if i < len(list_output) - 1:
                            output.write(str(list_output[i]) + " ")
                        else:
                            output.write(str(list_output[i]) + "\n\n")
            print("SVR", list_output)

        if way == 2:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(RFT(data, month)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            if tocsv:
                with open("../result_revision/mon{}_g{}_mt{}_{}.csv".format(month, goal, metrics, repo[:-12]),
                          "a+") as output:
                    output.write("RFT" + "\n")
                    for i in range(len(list_output)):
                        if i < len(list_output) - 1:
                            output.write(str(list_output[i]) + " ")
                        else:
                            output.write(str(list_output[i]) + "\n\n")
            print("RFT", list_output)

        if way == 3:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(CART(data, month)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            if tocsv:
                with open("../result_revision/mon{}_g{}_mt{}_{}.csv".format(month, goal, metrics, repo[:-12]),
                          "a+") as output:
                    output.write("CART" + "\n")
                    for i in range(len(list_output)):
                        if i < len(list_output) - 1:
                            output.write(str(list_output[i]) + " ")
                        else:
                            output.write(str(list_output[i]) + "\n\n")
            print("CART", list_output)

        if way == 4:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(KNN(data, month)[metrics])
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            if tocsv:
                with open("../result_revision/mon{}_g{}_mt{}_{}.csv".format(month, goal, metrics, repo[:-12]),
                          "a+") as output:
                    output.write("KNN" + "\n")
                    for i in range(len(list_output)):
                        if i < len(list_output) - 1:
                            output.write(str(list_output[i]) + " ")
                        else:
                            output.write(str(list_output[i]) + "\n\n")
            print("KNN", list_output)

        if way == 5:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(DECART(data, metrics, month))
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            if tocsv:
                with open("../result_revision/mon{}_g{}_mt{}_{}.csv".format(month, goal, metrics, repo[:-12]),
                          "a+") as output:
                    output.write("DECART" + "\n")
                    for i in range(len(list_output)):
                        if i < len(list_output) - 1:
                            output.write(str(list_output[i]) + " ")
                        else:
                            output.write(str(list_output[i]) + "\n\n")
            print("DECART", list_output)

        if way == 6:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(RDCART(data, metrics, month))
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            if tocsv:
                with open("../result_revision/mon{}_g{}_mt{}_{}.csv".format(month, goal, metrics, repo[:-12]),
                          "a+") as output:
                    output.write("RDCART" + "\n")
                    for i in range(len(list_output)):
                        if i < len(list_output) - 1:
                            output.write(str(list_output[i]) + " ")
                        else:
                            output.write(str(list_output[i]) + "\n\n")
            print("RDCART", list_output)

        if way == 7:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(DEKNN(data, metrics, month))
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            if tocsv:
                with open("../result_revision/mon{}_g{}_mt{}_{}.csv".format(month, goal, metrics, repo[:-12]),
                          "a+") as output:
                    output.write("DEKNN" + "\n")
                    for i in range(len(list_output)):
                        if i < len(list_output) - 1:
                            output.write(str(list_output[i]) + " ")
                        else:
                            output.write(str(list_output[i]) + "\n\n")
            print("DEKNN", list_output)

        if way == 8:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(RDKNN(data, metrics, month))
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            if tocsv:
                with open("../result_revision/mon{}_g{}_mt{}_{}.csv".format(month, goal, metrics, repo[:-12]),
                          "a+") as output:
                    output.write("RDKNN" + "\n")
                    for i in range(len(list_output)):
                        if i < len(list_output) - 1:
                            output.write(str(list_output[i]) + " ")
                        else:
                            output.write(str(list_output[i]) + "\n\n")
            print("RDKNN", list_output)

        if way == 9:
            list_temp = []
            for _ in range(repeats):
                list_temp.append(GSCART(data, metrics, month))
            flat_list = np.array(list_temp).flatten()
            list_output = sorted(flat_list.tolist())
            if tocsv:
                with open("../result_revision/mon{}_g{}_mt{}_{}.csv".format(month, goal, metrics, repo[:-12]),
                          "a+") as output:
                    output.write("GSCART" + "\n")
                    for i in range(len(list_output)):
                        if i < len(list_output) - 1:
                            output.write(str(list_output[i]) + " ")
                        else:
                            output.write(str(list_output[i]) + "\n\n")
            print("GSCART", list_output)


if __name__ == '__main__':

    path = r'../data/data_cleaned_sample/'
    # repo = "betaflight_monthly.csv"

    # goal == 0: 'number_of_commits'
    # goal == 1: 'number_of_contributors'
    # goal == 2: 'number_of_open_PRs'
    # goal == 3: 'number_of_closed_PRs'
    # goal == 4: 'number_of_open_issues'
    # goal == 5: 'number_of_closed_issues'
    # goal == 6: 'number_of_stargazers'
    # goal == 7: 'number_of_new_contributors'
    # goal == 8: 'number_of_contributor-domains'
    # goal == 9: 'number_of_new_contributor-domains'

    repo_pool = []
    # path = r'../temp_data/'
    for filename in os.listdir(path):
        if not filename.startswith('.'):
            repo_pool.append(os.path.join(filename))

    # Metrics = 0  # "0" for MRE, "1" for SA
    Repeat = 20

    time_begin = time.time()

    # number = 0
    # for mon in [1, 3, 6, 12]:
    #     for i in range(7):
    #         for mt in [0, 1]:
    #             for repo in sorted(repo_pool):
    #                 print("-----------------------------------------")
    #                 print(number, repo, "Metrics:", mt, "Goal:", i, "Month:", mon)
    #                 revision(repo, path, metrics=mt, repeats=Repeat, goal=i, month=mon, tocsv=True)
    #                 number += 1
    #
    # run_time = str(time.time() - time_begin)
    # print(run_time)

    sorted_repo_pool = sorted(repo_pool)

    number = 0
    for mon in [1, 3, 6, 12]:
        for i in range(7):
            for mt in [0, 1]:
                for repo in sorted_repo_pool:
                    print("-----------------------------------------")
                    print(number, repo, "Metrics:", mt, "Goal:", i, "Month:", mon)
                    revision(repo, path, metrics=mt, repeats=Repeat, goal=i, month=mon, tocsv=True)
                    number += 1

    run_time = str(time.time() - time_begin)
    print(run_time)

