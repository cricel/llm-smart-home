/* 
 * This message is auto generated by ROS#. Please DO NOT modify.
 * Note:
 * - Comments from the original code will be written in their own line 
 * - Variable sized arrays will be initialized to array of size 0 
 * Please report any issues at 
 * <https://github.com/siemens/ros-sharp> 
 */

#if ROS2

namespace RosSharp.RosBridgeClient.MessageTypes.Rosapi
{
    public class GetParam_Request : Message
    {
        public const string RosMessageName = "rosapi_msgs/msg/GetParam_Request";

        public string name { get; set; }
        public string default_value { get; set; }

        public GetParam_Request()
        {
            this.name = "";
            this.default_value = "";
        }

        public GetParam_Request(string name, string default_value)
        {
            this.name = name;
            this.default_value = default_value;
        }
    }
}
#endif
