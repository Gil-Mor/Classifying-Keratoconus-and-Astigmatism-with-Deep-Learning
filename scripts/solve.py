
from export_env_variables import *
from defs import *
from utils import *
import utils


# def mid_training_prediction_from_val_txt_with_deploy(caffe, deploy_net, transformer, imgs_file, data_dir, mean_binaryproto):
#
#     net = deploy_net
#
#     # ### 3. CPU classification
#     #
#     # * Now we're ready to perform classification. Even though we'll only classify one image, we'll set a batch size of 50 to demonstrate batching.
#
#     # In[6]:
#
#     # set the size of the input (we can skip this if we're happy
#     #  with the default; we can also change it later, e.g., for different batch sizes)
#     # !!!!!!!!!!!!!!1 This only made it predict 32 time for the same image
#     # net.blobs['data'].reshape(32,  # batch size
#     #                           3,  # 3-channel (BGR) images
#     #                           CLASSIFICATION_IMAGE_SIZE, CLASSIFICATION_IMAGE_SIZE)  # image size is 227x227
#
#     correct = 0
#     count = 1
#     # in my case, I'm reading the images file names from val.txt.
#     with open(imgs_file, "r") as f:
#         val_images = f.readlines()
#
#     print("Net blobs Data shape (transformed image should correspond in chan, height, width) ", net.blobs['data'].data.shape)
#
#     for image_name_n_label in val_images:
#         if len(image_name_n_label.split(' ')) != 2:
#             continue
#
#         # in val.txt every line contains - filename label
#         image_file, label = data_dir + "/" + image_name_n_label.split(' ')[0], int(image_name_n_label.split(' ')[1])
#
#         image = caffe.io.load_image(image_file)
#
#
#         # image shape is (3, 256, 256). we want it (3, 227, 227) for caffenet.
#         # asking about shape[0] and shape[1] because I can't know if the image is (chan, h, w) or (h, w, chan)
#         if image.shape[0] == TRAINING_IMAGE_SIZE or image.shape[1] == TRAINING_IMAGE_SIZE or image.shape[2] == TRAINING_IMAGE_SIZE:
#             # I'm cropping the numpy array on the fly so that I don't have to mess with resizing
#             # the actual images in a separate folder each time.
#             image = center_crop_image(image, CLASSIFICATION_IMAGE_SIZE, CLASSIFICATION_IMAGE_SIZE)
#
#
#         try:
#             transformed_image = transformer.preprocess('data', image)
#         except:
#             # try to transpose and again
#             image = image.transpose(2,0,1) # (height, width, chan) -> (chan, height, width)
#             transformed_image = transformer.preprocess('data', image)
#
#
#         # copy the image data into the memory allocated for the net
#         net.blobs['data'].data[...] = transformed_image
#
#         ### perform classification
#         output = net.forward(start='conv1')
#
#         output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
#         max_prob = max(output_prob)
#
#         predicted_label = output_prob.argmax()
#
#         if predicted_label == label:
#             correct += 1
#
#         print(str(count) + " " + image_file + " " + str(label) + " " + str(predicted_label) + " " + "{:.2f}".format(max_prob))
#         count += 1
#
#         # misclassified.close()
#
# # -------------------------------------------------------------------------------------------------------

def get_image_n_label_from_blob(net, image_index):
    return net.blobs['data'].data[image_index].copy(), np.array(net.blobs['label'].data, dtype=np.int32)[image_index],
# -------------------------------------------------------------------------------------------------------


def predict_for_one_image_using_test_net(caffe, net, image, ground_truth, labels, num_of_classes):


    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['prob'][0]


    top_k = (-probs).argsort()[:num_of_classes]

    print("ground truth: {}".format(labels[ground_truth]))
    for i,p in enumerate(top_k):
        print("{}. label: {:<20}.  pred {:.2f}".format(i, labels[p], 100*probs))
    print("")
    if (top_k[0] == ground_truth):
        print("success")
        return True
    else:
        print("fail")
        return False

# -------------------------------------------------------------------------------------------------------

def solve(iterations, solver_prototxt, weights, display_iter, test_iter, val_txt, mean_binaryproto, data_dir, solverstate=None):
    """Run solvers for niter iterations,
           returning the loss and accuracy recorded each iteration.
           `solvers` is a list of (name, solver) tuples."""
    # print(val_txt)
    # with open(val_txt) as f:
    #     lines = f.readlines()
    # num_of_val_imgs = len([line for line in lines if line != ""])
    # print(num_of_val_imgs)
    # return

    sys.path.append(pycaffe_module_path)
    caffe = import_caffe()

    # g_log = open(mode.log.replace(".log", "_manual.log") , "w")
    # mode.state = "pycaffe"

    solver = caffe.get_solver(solver_prototxt)
    if solverstate is not None:
        solver.restore(solverstate)

    print("using weights ", os.path.basename(weights))
    solver.net.copy_from(weights)
    print("starting from iteration ", solver.iter)

    train_loss, val_loss, acc = [], [], []


    for _ in range(iterations):

        if solver.iter % display_iter == 0:
            train_loss.append(solver.net.blobs['loss'].data.copy())



        image, label = get_image_n_label_from_blob(solver.net, 0)
        print("image from data\n",image)
        print(label)

        if solver.iter % test_iter == 0:
            # solver.test_nets[0].forward()
            # solver.net.forward()
            # out = solver.test_nets[0].forward()
            # print(out)
            # print(out['prob'])

            val_labels = list(solver.test_nets[0].blobs['label'].data.copy().astype(np.int))
            val_propabilities = solver.test_nets[0].blobs['prob'].data.copy()
            predicted = [tup.argmax() for tup in val_propabilities]
            print("labels      ", val_labels)
            print("predictions ", predicted)

            val_loss.append(solver.test_nets[0].blobs['loss'].data.copy())
            acc.append(solver.test_nets[0].blobs['accuracy'].data.copy())
            #
            # pred_label = np.array(solver.test_nets[0].blobs['loss'], dtype=np.int32)[0]
            #
            # image, label = get_image_n_label_from_blob(solver.test_nets[0], 0)

            # predict_for_one_image_using_test_net(caffe, solver.test_nets[0], image, label, ['healthy', 'kc'], num_of_classes=2)




            # filters = solver.net.params['conv1'][0].data
            # show_blobs(filters.transpose(0, 2, 3, 1))
            # feat = solver.net.blobs['conv2'].data[0, :36]
            # feat = solver.net.blobs['data'].data[0]
            # show_blobs(feat)
            # check_for_overfitting(loss, acc)

        # step here because we want to test in iteration 0 as well!!
        solver.step(1)  # run a single SGD step in Caffe


# -------------------------------------------------------------------------------------------------------

if __name__=="__main__":
    iterations = int(sys.argv[1])
    solver_prototxt = sys.argv[2]
    weights = sys.argv[3]
    display_iter = int(sys.argv[4])
    test_iter = int(sys.argv[5])
    val_txt = sys.argv[6]
    mean_binaryproto = sys.argv[7]
    data_dir = sys.argv[8]
    solve(iterations, solver_prototxt, weights, display_iter, test_iter, val_txt, mean_binaryproto, data_dir)