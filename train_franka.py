import numpy as np
from franka_gym import FrankaGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
import os

def train_franka_robot(
    total_timesteps=100000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    render_mode=None,
    save_dir="./models",
    use_eval_callback=False,
    resume_from=None
):
    """
    Train a Franka robot using PPO algorithm.
    
    Args:
        total_timesteps: Total training steps
        learning_rate: Learning rate for the algorithm
        n_steps: Number of steps per rollout
        batch_size: Batch size for training
        render_mode: Render mode ("human" or None)
        save_dir: Directory to save models
        use_eval_callback: Whether to use evaluation callback (disable if it hangs)
        resume_from: Path to existing model checkpoint to resume training from
    """
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print("Initializing environment...")
    
    # Create vectorized environment with render_mode=None for training (much faster)
    env = make_vec_env(
        lambda: FrankaGymEnv(render_mode=render_mode),
        n_envs=8 if render_mode is None else 1, 
    )
    
    # Create evaluation environment only if using eval callback
    eval_env = FrankaGymEnv(render_mode=render_mode) if use_eval_callback else None
    
    print("Setting up PPO agent...")
    
    if resume_from:
        print(f"Loading model from checkpoint: {resume_from}")
        model = PPO.load(resume_from, env=env)
        print(f"Resumed training from {model.num_timesteps} timesteps")
    else:
        # Initialize PPO agent
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            verbose=1,
            device="cuda",
            tensorboard_log="./tensorboard_logs",
        )
    
    # Set up callbacks
    callbacks = []
    
    if use_eval_callback:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            log_path=save_dir,
            eval_freq=5000,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=save_dir,
        name_prefix="franka_checkpoint",
    )
    callbacks.append(checkpoint_callback)
    
    print(f"Training for {total_timesteps} timesteps...")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save the final model
    final_model_path = os.path.join(save_dir, "franka_final_model")
    model.save(final_model_path)
    print(f"\nTraining complete! Model saved to {final_model_path}")
    
    # Close environments
    env.close()
    if eval_env is not None:
        eval_env.close()
    
    return model, final_model_path


def evaluate_trained_model(model_path, num_episodes=5, render_mode="human"):
    """
    Evaluate a trained model on the environment.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of episodes to evaluate
        render_mode: Render mode for visualization
    """
    print(f"\nLoading model from {model_path}...")
    model = PPO.load(model_path)
    
    env = FrankaGymEnv(render_mode=render_mode)
    
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        
        while not done and step_count < 200:
            # Use the trained policy to select action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step_count += 1
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.4f}, Steps = {step_count}")
    
    env.close()
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nAverage Reward: {avg_reward:.4f} ± {std_reward:.4f}")
    
    return total_rewards


if __name__ == "__main__":
    import argparse
    
    # parser = argparse.ArgumentParser(description="Train Franka robot with RL")
    # parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps")
    # parser.add_argument("--eval", action="store_true", help="Only evaluate existing model")
    # parser.add_argument("--model-path", type=str, default="./models/franka_final_model", help="Path to model for evaluation")
    # parser.add_argument("--render", action="store_true", help="Enable rendering during training")
    # parser.add_argument("--use-eval-callback", action="store_true", help="Use evaluation callback during training (disable if it hangs)")
    # parser.add_argument("--resume-from", type=str, help="Path to checkpoint model to resume training from")
    
    # args = parser.parse_args()
    args = type('Args', (), {
        "timesteps": 3000000, #2000000
        "eval": True,
        "model_path": "./models/franka_final_model",
        "render": False,
        "use_eval_callback": False,
        "resume_from": None,
    })()
    
    if args.eval:
        # Evaluate existing model
        evaluate_trained_model(args.model_path, num_episodes=20, render_mode="human")
    else:
        # Train new model
        model, model_path = train_franka_robot(
            total_timesteps=args.timesteps,
            render_mode="human" if args.render else None,
            use_eval_callback=args.use_eval_callback,
            resume_from=args.resume_from,
        )
        
        # Evaluate the trained model
        print("\n" + "="*50)
        print("Evaluating trained model...")
        print("="*50)
        evaluate_trained_model(model_path, num_episodes=3, render_mode="human")


## TODO: simplify the scene setting, just need to bring the cube from the 
# ground to the target, no need for the table