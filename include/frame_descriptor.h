#pragma once

#include <DBoW2/DBoW2.h>

//#include <opencv/cv.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv/cv.h>
#include <opencv2/nonfree/nonfree.hpp>

namespace slc {
    struct FrameDescriptor {

        std::unique_ptr<Lift128Vocabulary> vocab_;

        /*
        * Computes a global representation for an image by using
        * the SURF feature descriptor in OpenCV and the bag of
        * words approach.
        */

        FrameDescriptor(const std::string& vocabulary_path)
        {
            std::cout << "Loading vocabulary from " << vocabulary_path << std::endl;
            vocab_.reset(new Lift128Vocabulary(vocabulary_path));
            std::cout << "Loaded vocabulary with " << vocab_->size() << " visual words." << std::endl;
        }

        vector<float> InputData_To_Vector(string ss)
        {
            vector<float> p;
            ifstream infile(ss.c_str());
            float number;
            while(! infile.eof())
                {
                    infile >> number;
                    p.push_back(number);
                }
                p.pop_back();  //此处要将最后一个数字弹出，是因为上述循环将最后一个数字读取了两次
                return p;
        }

        void extract_surf(unsigned int i, std::vector<std::vector<float>>& descriptors) {

            std::stringstream ss;
            ss << "../images/desc_image" << i << ".txt";

            vector<float> plain = InputData_To_Vector(ss.str());

            const int L = 128;
            descriptors.resize(plain.size() / L);

            unsigned int j = 0;
            for (unsigned int i = 0; i < plain.size(); i += L, ++j) {
                descriptors[j].resize(L);
                std::copy(plain.begin() + i, plain.begin() + i + L, descriptors[j].begin());
                }
            }



        void describe_frame(unsigned int i, DBoW2::BowVector& bow_vec) {
            /* Transforms the feature descriptors to a BoW representation
            *  of the whole image. */

            //std::vector<cv::KeyPoint> keypoints;
            std::vector<std::vector<float>> descriptors;
            unsigned int file_num = i;

            extract_surf(file_num, descriptors);
            vocab_->transform(descriptors, bow_vec);

        }
        
    };   
}
