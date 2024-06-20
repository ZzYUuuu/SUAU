"""
@Time    : 2024/3/11 08:25
@Author  : YuZhang
@File    : trainer.py
"""
import torch
from time import time
import utility.tools as tools
import utility.tester as tester
import os
import datetime
def training(model, args, dataset, device, logger):
    model.to(device)

    best_recall_epoch, best_recall_1, best_ndcg_1 = 0, 0, 0
    cnt = 0
    optim = torch.optim.Adam(model.parameters(), lr=float(args.learn_rate))
    logger.info(model)
    logger.info(optim)
    # cold_dict, norm_dict, hot_dict = dataset.get_cold_norm_hot()

    for epoch in range(int(args.train_epoch)):
        start_time = time()
        total_loss_list = []

        model.train()

        user_pos_neg_pairs = dataset.create_user_pos()
        users = torch.Tensor(user_pos_neg_pairs[:, 0]).long()
        pos_items = torch.Tensor(user_pos_neg_pairs[:, 1]).long()

        # item_indices, similar_item_indices = dataset.hot_func2()
        item_indices, similar_item_indices = model.find_similar_item()
        item_indices, similar_item_indices = torch.Tensor(item_indices).long(), torch.Tensor(similar_item_indices).long()
        item_indices, similar_item_indices = item_indices.to(device), similar_item_indices.to(device)

        users = users.to(device)
        pos_items = pos_items.to(device)
        users, pos_items = tools.shuffle(users, pos_items)

        # main
        num_batch = len(users) // int(args.train_batch_size) + 1
        for batch_i, (batch_user, batch_pos) in \
                enumerate(tools.mini_batch(users, pos_items, batch_size=int(args.train_batch_size))):

            loss_list = model(batch_user, batch_pos)

            total_loss = 0.

            if batch_i == 0:
                assert len(loss_list) > 1
                total_loss_list = [0.] * len(loss_list)

            for i in range(len(loss_list)):
                loss = loss_list[i]
                total_loss += loss
                total_loss_list[i] += loss.item()
            print('\t Step %d/%d: loss = %.8f' % (batch_i, num_batch, total_loss), end='\r')
            optim.zero_grad()
            total_loss.backward()
            optim.step()

        # item
        num_batch1 = len(item_indices) // int(args.train_batch_size) + 1
        for batch_i, (batch_item1, batch_item2) in \
                enumerate(tools.mini_batch(item_indices, similar_item_indices, batch_size=int(args.train_batch_size))):
            loss_list = model(batch_item1, batch_item2, flag='item')
            total_loss = 0.
            if batch_i == 0:
                assert len(loss_list) > 1
                total_loss_list_item = [0.] * len(loss_list)

            for i in range(len(loss_list)):
                loss = loss_list[i]
                total_loss += loss
                total_loss_list_item[i] += loss.item()
            print('\t Step %d/%d: loss = %.8f' % (batch_i, num_batch1, total_loss), end='\r')
            optim.zero_grad()
            total_loss.backward()
            optim.step()

        end_time = time()
        loss = round(sum(total_loss_list) / num_batch, 6)
        loss_strs = str(loss) + "=" + "+".join([str(round(i / num_batch, 6)) for i in total_loss_list])

        #get loss
        # result_list = [item / num_batch for item in total_loss_list]
        # str1 = './log/UniqueAU/'+str(args.dataset)+'/loss.txt'
        # with open(str1, 'a') as file:
        #     file.write(' '.join(map(str, result_list)) + '\n')
        #-------

        print("\t Epoch: %4d| train time %.3f| train loss: %s" % (epoch + 1, end_time - start_time, loss_strs))
        logger.info("\t Epoch: %4d| train time %.3f| train loss: %s" % (epoch + 1, end_time - start_time, loss_strs))
        if epoch % int(args.test_frequency) == 0:
            if args.sparsity_test == 0:
                model_results = tester.testing(model, args, dataset, device)
                cnt += 1
                if model_results['recall'][1] > best_recall_1:
                    cnt = 0
                    best_recall_epoch = epoch + 1
                    best_recall_1 = model_results['recall'][1]
                    best_ndcg_1 = model_results['ndcg'][1]
                    # best_model_state_dict = model.state_dict()
                print("\t Recall:" + str(model_results['recall']) + "\n\t NDCG:  " + str(model_results['ndcg']))
                logger.info(
                    "\t Recall:" + str(model_results['recall']) + "\n" + "\t" * 7 + " NDCG:  " + str(model_results['ndcg']))
            else:
                result = tester.sparsity_test(dataset, args, model, device)
                if result[0]['recall'][0] > best_recall_1:
                    best_report_epoch = epoch + 1
                    best_report_recall = result[0]['recall'][0]
                print("\t level_1: recall:", result[0]['recall'], ',ndcg:',
                      result[0]['ndcg'])
                print("\t level_2: recall:", result[1]['recall'], ',ndcg:',
                      result[1]['ndcg'])
                print("\t level_3: recall:", result[2]['recall'], ',ndcg:',
                      result[2]['ndcg'])
                print("\t level_4: recall:", result[3]['recall'], ',ndcg:',
                      result[3]['ndcg'])
                logger.info("\t level_1: recall:" + str(result[0]['recall']) + ',ndcg:' + str(result[0]['ndcg']))
                logger.info("\t level_2: recall:" + str(result[1]['recall']) + ',ndcg:' + str(result[1]['ndcg']))
                logger.info("\t level_3: recall:" + str(result[2]['recall']) + ',ndcg:' + str(result[2]['ndcg']))
                logger.info("\t level_4: recall:" + str(result[3]['recall']) + ',ndcg:' + str(result[3]['ndcg']))


            # ----------------------test group
            # model_results = tester.testing_group(model, args, dataset, device, cold_dict)
            # print("\t Recall:" + str(model_results['recall']) + "\n\t NDCG:  " + str(model_results['ndcg']))
            # logger.info(
            #     "\t Recall:" + str(model_results['recall']) + "\n" + "\t" * 7 + " NDCG:  " + str(model_results['ndcg']))
            # model_results = tester.testing_group(model, args, dataset, device, norm_dict)
            # print("\t Recall:" + str(model_results['recall']) + "\n\t NDCG:  " + str(model_results['ndcg']))
            # logger.info(
            #     "\t Recall:" + str(model_results['recall']) + "\n" + "\t" * 7 + " NDCG:  " + str(model_results['ndcg']))
            # model_results = tester.testing_group(model, args, dataset, device, hot_dict)
            # print("\t Recall:" + str(model_results['recall']) + "\n\t NDCG:  " + str(model_results['ndcg']))
            # logger.info(
            #     "\t Recall:" + str(model_results['recall']) + "\n" + "\t" * 7 + " NDCG:  " + str(model_results['ndcg']))



            if cnt > int(args.early_stop):
                break
    # torch.save(best_model_state_dict, '../log/UniqueAU/'+str(args.dataset)+'/best_model_state_dict.pth')
    print("\t Model training process completed.")
    print("\t best recall epoch:" + str(best_recall_epoch))
    print("\t best recall:" + str(best_recall_1) + "\t best ndcg:" + str(best_ndcg_1))

    logger.info("\t Model training process completed.")
    logger.info("\t best recall epoch:" + str(best_recall_epoch))
    logger.info("\t best recall:" + str(best_recall_1) + "\t best ndcg:" + str(best_ndcg_1))

    """
    save file to result folder
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_folder = "results"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    file_path = os.path.join(result_folder, f"{model.model_name}_results_{current_time}.txt")
    with open(file_path, 'w') as file:
        file.write("Model Training Results for {}:\n".format(model.model_name))
        # file.write(completed_message + "\n")
        file.write("Best Recall Epoch: " + str(best_recall_epoch) + "\n")
        file.write("Best Recall: " + str(best_recall_1) + "\n")
        file.write("Best NDCG: " + str(best_ndcg_1) + "\n")
        file.write("File created at: " + current_time + "\n")

    handlers = logger.handlers

    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()