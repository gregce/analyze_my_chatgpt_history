"""
Parser for ChatGPT conversation history.
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

def parse_chatgpt_conversations(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Parse ChatGPT conversation history from a JSON file into a pandas DataFrame.
    
    Args:
        file_path: Path to the conversations.json file
        
    Returns:
        DataFrame containing parsed conversations with columns:
        - conversation_id: Unique identifier for each conversation
        - conversation_title: Title of the conversation
        - message_id: Unique identifier for each message
        - parent_id: ID of the parent message (for threading)
        - create_time: Timestamp when message was created
        - author_role: Role of message author (user/assistant/system)
        - content: The actual message content
        - status: Message status
        - model: The model used for assistant responses
    """
    # Read the JSON file
    file_path = Path(file_path)
    with file_path.open('r') as f:
        conversations = json.load(f)
    
    # List to store flattened messages
    messages: List[Dict] = []
    
    for conv in conversations:
        title = conv['title']
        conv_id = conv['conversation_id']
        
        # Process each message in the mapping
        for msg_id, msg_data in conv['mapping'].items():
            if msg_data.get('message'):  # Some mapping entries might not have messages
                msg = msg_data['message']
                
                # Handle create_time being None
                create_time = msg.get('create_time')
                if create_time is not None:
                    create_time = datetime.fromtimestamp(create_time)
                
                # Extract relevant fields
                message_dict = {
                    'conversation_id': conv_id,
                    'conversation_title': title,
                    'message_id': msg_id,
                    'parent_id': msg_data.get('parent'),
                    'create_time': create_time,
                    'author_role': msg['author'].get('role'),
                    'content': msg['content'].get('parts', [''])[0] if msg['content'].get('parts') else '',
                    'status': msg.get('status'),
                    'model': msg.get('metadata', {}).get('model_slug')
                }
                
                messages.append(message_dict)
    
    # Create DataFrame and sort by timestamp
    df = pd.DataFrame(messages)
    
    # Sort by create_time, handling None values
    df = df.sort_values('create_time', na_position='first')
    
    return df

def analyze_conversations(df: pd.DataFrame) -> Dict:
    """
    Perform basic analysis on the conversation DataFrame.
    
    Args:
        df: DataFrame from parse_chatgpt_conversations()
        
    Returns:
        Dictionary containing analysis results:
        - total_messages: Total number of messages
        - total_conversations: Total number of unique conversations
        - messages_by_role: Count of messages by author role
        - models_used: Count of different models used
        - avg_messages_per_conversation: Average messages per conversation
        - messages_with_timestamp: Number of messages with valid timestamps
    """
    analysis = {
        'total_messages': len(df),
        'total_conversations': df['conversation_id'].nunique(),
        'messages_by_role': df['author_role'].value_counts().to_dict(),
        'models_used': df['model'].value_counts().to_dict(),
        'avg_messages_per_conversation': len(df) / df['conversation_id'].nunique(),
        'messages_with_timestamp': df['create_time'].notna().sum()
    }
    
    return analysis

def format_date_xaxis(ax, dates):
    """Helper function to format date axis nicely."""
    # Determine date range
    date_range = (max(dates) - min(dates)).days
    
    if date_range <= 60:  # Less than 2 months
        ax.xaxis.set_major_locator(mdates.WeekLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif date_range <= 180:  # Less than 6 months
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:  # More than 6 months
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

def plot_chat_frequency(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create a bar plot showing conversation frequency by week.
    
    Args:
        df: DataFrame from parse_chatgpt_conversations()
        output_path: Path to save the plot
    """
    # Filter out rows without timestamps
    df_with_time = df[df['create_time'].notna()].copy()
    
    # Convert to datetime and extract week
    df_with_time['week_start'] = pd.to_datetime(df_with_time['create_time']).dt.to_period('W').dt.start_time
    
    # Count conversations per week
    weekly_counts = df_with_time.groupby('week_start')['conversation_id'].nunique().reset_index()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    # Calculate bar width based on data range
    time_range = (weekly_counts['week_start'].max() - weekly_counts['week_start'].min()).days
    num_bars = len(weekly_counts)
    width = max(time_range / (num_bars * 2), 4)  # Ensure minimum width of 4 days
    
    # Create bar plot with calculated width
    plt.bar(weekly_counts['week_start'], weekly_counts['conversation_id'], 
            alpha=0.7, color=sns.color_palette("husl", 8)[0],
            width=pd.Timedelta(days=width))
    
    # Format x-axis
    format_date_xaxis(ax, weekly_counts['week_start'])
    
    # Customize the plot
    plt.title('ChatGPT Conversations Per Week', fontsize=14, pad=20)
    plt.xlabel('Week Starting', fontsize=12)
    plt.ylabel('Number of Conversations', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path / 'chat_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_message_frequency(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create a bar plot showing message frequency by week.
    
    Args:
        df: DataFrame from parse_chatgpt_conversations()
        output_path: Path to save the plot
    """
    # Filter out rows without timestamps
    df_with_time = df[df['create_time'].notna()].copy()
    
    # Convert to datetime and extract week
    df_with_time['week_start'] = pd.to_datetime(df_with_time['create_time']).dt.to_period('W').dt.start_time
    
    # Count messages per week
    weekly_counts = df_with_time.groupby('week_start').size().reset_index(name='message_count')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    # Calculate bar width based on data range
    time_range = (weekly_counts['week_start'].max() - weekly_counts['week_start'].min()).days
    num_bars = len(weekly_counts)
    width = max(time_range / (num_bars * 2), 4)  # Ensure minimum width of 4 days
    
    # Create bar plot with calculated width
    plt.bar(weekly_counts['week_start'], weekly_counts['message_count'], 
            alpha=0.7, color=sns.color_palette("husl", 8)[1],  # Use a different color
            width=pd.Timedelta(days=width))
    
    # Format x-axis
    format_date_xaxis(ax, weekly_counts['week_start'])
    
    # Customize the plot
    plt.title('ChatGPT Messages Per Week', fontsize=14, pad=20)
    plt.xlabel('Week Starting', fontsize=12)
    plt.ylabel('Number of Messages', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path / 'message_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_monthly_messages(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create a bar plot showing total messages per month.
    
    Args:
        df: DataFrame from parse_chatgpt_conversations()
        output_path: Path to save the plot
    """
    # Filter out rows without timestamps
    df_with_time = df[df['create_time'].notna()].copy()
    
    # Convert to datetime and extract month
    df_with_time['month'] = pd.to_datetime(df_with_time['create_time']).dt.to_period('M').dt.start_time
    
    # Count messages per month
    monthly_counts = df_with_time.groupby('month').size().reset_index(name='message_count')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    # Create bar plot with fixed width for monthly data
    plt.bar(monthly_counts['month'], monthly_counts['message_count'], 
            alpha=0.7, color=sns.color_palette("husl", 8)[2],  # Use a different color
            width=25)  # Width of approximately 25 days for monthly bars
    
    # Format x-axis
    format_date_xaxis(ax, monthly_counts['month'])
    
    # Add value labels on top of each bar
    for i, v in enumerate(monthly_counts['message_count']):
        ax.text(monthly_counts['month'].iloc[i], v, str(v), 
                ha='center', va='bottom')
    
    # Customize the plot
    plt.title('ChatGPT Messages Per Month', fontsize=14, pad=20)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Messages', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path / 'monthly_messages.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_usage(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create line plot showing model usage over time and save model distribution as a table.
    
    Args:
        df: DataFrame from parse_chatgpt_conversations()
        output_path: Path to save the plots
    """
    # Filter out rows without timestamps or models
    df_with_time = df[df['create_time'].notna() & df['model'].notna()].copy()
    
    if len(df_with_time) == 0:
        return  # No data to plot
    
    # Convert to datetime and extract week
    df_with_time['week_start'] = pd.to_datetime(df_with_time['create_time']).dt.to_period('W').dt.start_time
    
    # Count models used per week
    model_counts = df_with_time.groupby(['week_start', 'model']).size().unstack(fill_value=0)
    
    # Create the line plot
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    # Plot lines for each model
    for column in model_counts.columns:
        plt.plot(model_counts.index, model_counts[column], marker='o', label=column, linewidth=2)
    
    # Format x-axis
    format_date_xaxis(ax, model_counts.index)
    
    # Customize the plot
    plt.title('ChatGPT Model Usage Over Time', fontsize=14, pad=20)
    plt.xlabel('Week Starting', fontsize=12)
    plt.ylabel('Number of Messages', fontsize=12)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the line plot
    plt.savefig(output_path / 'model_usage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create and save model distribution as a table
    total_model_usage = df_with_time['model'].value_counts()
    total_messages = len(df_with_time)
    
    # Create DataFrame with counts and percentages
    model_stats = pd.DataFrame({
        'Messages': total_model_usage,
        'Percentage': (total_model_usage / total_messages * 100).round(1)
    })
    model_stats['Percentage'] = model_stats['Percentage'].map('{:.1f}%'.format)
    
    # Save the table to a text file
    with open(output_path / 'model_distribution.txt', 'w') as f:
        f.write("Model Usage Distribution\n")
        f.write("=" * 40 + "\n\n")
        f.write(model_stats.to_string())
        f.write("\n\nTotal Messages: " + str(total_messages))