import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Path


class MultipleGoalPublisher(Node):

    def __init__(self):
        super().__init__('multiple_goal_publisher')

        # Subscribe to the clicked point topic
        self.subscription = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.point_callback,
            10)

        # Publisher to send goals to the Nav2 stack
        self.goal_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Store multiple goals in a list
        self.goals = []

        # Create a timer to publish goals periodically
        self.timer = self.create_timer(2.0, self.publish_goal)

    def point_callback(self, msg):
        # Create a PoseStamped message from the clicked point
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position = msg.point
        goal_pose.pose.orientation.w = 1.0  # Default orientation

        # Store the goal
        self.goals.append(goal_pose)
        self.get_logger().info(f'Added goal: {msg.point}')

    def publish_goal(self):
        if self.goals:
            # Publish the first goal from the list
            current_goal = self.goals.pop(0)
            self.goal_publisher.publish(current_goal)
            self.get_logger().info(f'Published goal: {current_goal.pose.position}')
        else:
            self.get_logger().info('No more goals to publish.')


def main(args=None):
    rclpy.init(args=args)
    multiple_goal_publisher = MultipleGoalPublisher()
    rclpy.spin(multiple_goal_publisher)
    multiple_goal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
