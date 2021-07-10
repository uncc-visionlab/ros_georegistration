#include <iostream>
#include <memory>
#include <math.h>
//#include <deque>
//#include <sdf/sdf.hh>

#include "gazebo/gazebo.hh"
#include "gazebo/common/common.hh"
#include "gazebo/transport/transport.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/msgs/msgs.hh"

#include "model_bbox_request.pb.h"
#include "model_bbox_response.pb.h"

bool noMsgRcvd = true;

namespace gazebo {

    typedef const boost::shared_ptr<const model_bbox_msgs::msgs::ModelBoundingBoxResponse>
    ModelBoundingBoxResponsePtr;
    
    void receiveModelSize(ModelBoundingBoxResponsePtr &res) {
        std::cout << "Received model size (" 
                << res->size().x() << ", "
                << res->size().y() << ", " 
                << res->size().z() << ") "  << std::endl;
        std::cout << "Received model size (" 
                << res->scale().x() << ", "
                << res->scale().y() << ", " 
                << res->scale().z() << ") "  << std::endl;
        noMsgRcvd = false;
    }
}

int main(int argc, char * argv[])
{
    if (argc > 2)
    {
        model_bbox_msgs::msgs::ModelBoundingBoxRequest request;
        request.set_model_name(argv[1]);
        request.set_link_name(argv[2]);

        gazebo::transport::init();
        gazebo::transport::run();
        gazebo::transport::NodePtr node(new gazebo::transport::Node());
        node->Init("default");

        std::cout << "Subscribing to: " << "~/modelsize/response" << std::endl;
        gazebo::transport::SubscriberPtr commandSubscriber = node->Subscribe("~/modelsize/response",
                &gazebo::receiveModelSize);
        std::cout << "Request: " <<
                "model_name: " << request.model_name() << ", "
                "link_name: " << request.link_name() << std::endl;
        std::cout << "Publishing to: " << "~/modelsize/request" << std::endl;
        gazebo::transport::PublisherPtr imagePub =
                node->Advertise<model_bbox_msgs::msgs::ModelBoundingBoxRequest>(
                                                            "~/modelsize/request");
        imagePub->WaitForConnection();
        imagePub->Publish(request);
        std::cout << "Published request" << std::endl;
        while(noMsgRcvd) {
            
        }
        std::cout << "Response received" << std::endl;
        gazebo::transport::fini();
        return 0;
    }
    return -1;
}