/* 
 * This message is auto generated by ROS#. Please DO NOT modify.
 * Note:
 * - Comments from the original code will be written in their own line 
 * - Variable sized arrays will be initialized to array of size 0 
 * Please report any issues at 
 * <https://github.com/siemens/ros-sharp> 
 */

#if ROS2

namespace RosSharp.RosBridgeClient.MessageTypes.Moveit
{
    public class TrajectoryConstraints : Message
    {
        public const string RosMessageName = "moveit_msgs/msg/TrajectoryConstraints";

        //  The array of constraints to consider along the trajectory
        public Constraints[] constraints { get; set; }

        public TrajectoryConstraints()
        {
            this.constraints = new Constraints[0];
        }

        public TrajectoryConstraints(Constraints[] constraints)
        {
            this.constraints = constraints;
        }
    }
}
#endif
