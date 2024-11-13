/* 
 * This message is auto generated by ROS#. Please DO NOT modify.
 * Note:
 * - Comments from the original code will be written in their own line 
 * - Variable sized arrays will be initialized to array of size 0 
 * Please report any issues at 
 * <https://github.com/siemens/ros-sharp> 
 */

#if !ROS2
using RosSharp.RosBridgeClient.MessageTypes.Std;
using RosSharp.RosBridgeClient.MessageTypes.Actionlib;

namespace RosSharp.RosBridgeClient.MessageTypes.ActionlibTutorials
{
    public class AveragingActionResult : ActionResult<AveragingResult>
    {
        public const string RosMessageName = "actionlib_tutorials/AveragingActionResult";

        public AveragingActionResult() : base()
        {
            this.result = new AveragingResult();
        }

        public AveragingActionResult(Header header, GoalStatus status, AveragingResult result) : base(header, status)
        {
            this.result = result;
        }
    }
}
#endif
