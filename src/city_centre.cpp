#include <vector>
#include <utility>
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <cstdlib>

#include <DBoW2/DBoW2.h>

#include "frame_descriptor.h"
//#include "utils.h"

using namespace std;
using namespace slc;


int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage " << argv[0] << " [path_to_vocabulary] " << endl;
        exit(1);
    }
    cout << "put citycentre lift descriptor .txt files in /citycentre_loop_closure documentary" << endl;

    string vocabulary_path(argv[1]);
    FrameDescriptor descriptor(vocabulary_path);

    //string dataset_folder(argv[2]); [path_to_IJRR_2008_images]
    //auto filenames = load_filenames(dataset_folder);
    //std::cout << "Processing " << filenames.size() << " images\n";

    // Will hold BoW representations for each frame
    vector<DBoW2::BowVector> bow_vecs;

    for (unsigned int img_i = 1; img_i < 2475; img_i++) {

        // Get a BoW description of the current image
        DBoW2::BowVector bow_vec;
        //descriptor.describe_frame(img_i, bow_vec);
        descriptor.describe_frame(img_i, bow_vec);
        bow_vecs.push_back(bow_vec);
        //descriptor.describe_frame(2, bow_vec);
        //bow_vecs.push_back(bow_vec);
    }

    cout << "Writing output..." << endl;

    //cout << "frame 1 to vector:" << bow_vec << endl;

    /*ofstream of;
    of.open(
        getenv("HOME") + string("/dev/simple_slam_loop_closure/out/confusion_matrix.txt"));*/

    ofstream of;
    of.open(
        getenv("HOME") + string("/citycentre_loop_closure/out/citycentrelift_matrix.txt"));

    // Compute confusion matrix
    // i.e. the (i, j) element of the matrix contains the distance
    // between the BoW representation of frames i and j
    for (unsigned int i = 0; i < bow_vecs.size(); i++) {
       for (unsigned int j = 0; j < bow_vecs.size(); j++) {
            of << descriptor.vocab_->score(
                bow_vecs[i], bow_vecs[j]) << " ";
        }
        of << "\n";
    }

    of.close();
    cout << "Output done" << endl;
}
