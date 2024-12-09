"""
Main script for analyzing ChatGPT conversations.
"""

import sys
from datetime import datetime
from pathlib import Path
from .parser import (
    parse_chatgpt_conversations, 
    analyze_conversations, 
    plot_chat_frequency, 
    plot_message_frequency,
    plot_monthly_messages,
    plot_model_usage
)

def create_output_directory() -> Path:
    """Create a timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"chat_analysis_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def save_analysis_results(analysis: dict, df, output_dir: Path) -> None:
    """Save analysis results to a text file."""
    with open(output_dir / "analysis_results.txt", "w") as f:
        f.write("ChatGPT Conversation Analysis Results\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Total Messages: {analysis['total_messages']}\n")
        f.write(f"Messages with Timestamps: {analysis['messages_with_timestamp']}\n")
        f.write(f"Total Conversations: {analysis['total_conversations']}\n")
        f.write(f"Average Messages per Conversation: {analysis['avg_messages_per_conversation']:.2f}\n\n")
        
        f.write("Messages by Role:\n")
        for role, count in analysis['messages_by_role'].items():
            f.write(f"  {role}: {count}\n")
        f.write("\n")
        
        f.write("Models Used:\n")
        for model, count in analysis['models_used'].items():
            if model:  # Only show non-null models
                f.write(f"  {model}: {count}\n")
        f.write("\n")
        
        if analysis['messages_with_timestamp'] > 0:
            first_date = df[df['create_time'].notna()]['create_time'].min()
            last_date = df[df['create_time'].notna()]['create_time'].max()
            f.write("Conversation Date Range:\n")
            f.write(f"  First Message: {first_date}\n")
            f.write(f"  Last Message: {last_date}\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m chatgpt_analysis <path_to_conversations.json>")
        sys.exit(1)
        
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"\nCreated output directory: {output_dir}")
        
    # Parse conversations
    print(f"Parsing conversations from {file_path}...")
    df = parse_chatgpt_conversations(file_path)
    
    # Analyze conversations
    print("Analyzing conversations...")
    analysis = analyze_conversations(df)
    
    # Save results
    print("Saving analysis results...")
    save_analysis_results(analysis, df, output_dir)
    
    # Create and save visualizations
    print("Generating visualizations...")
    plot_chat_frequency(df, output_dir)
    plot_message_frequency(df, output_dir)
    plot_monthly_messages(df, output_dir)
    plot_model_usage(df, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"- Text results: {output_dir}/analysis_results.txt")
    print(f"- Weekly conversation frequency: {output_dir}/chat_frequency.png")
    print(f"- Weekly message frequency: {output_dir}/message_frequency.png")
    print(f"- Monthly message count: {output_dir}/monthly_messages.png")
    print(f"- Model usage over time: {output_dir}/model_usage.png")
    print(f"- Model distribution: {output_dir}/model_distribution.txt")

if __name__ == "__main__":
    main() 