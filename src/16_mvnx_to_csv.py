import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

def get_sensor_names(root: ET.Element) -> List[str]:
    """
    Extract sensor names from the MVNX file structure.
    
    Args:
        root: Root element of the MVNX XML tree
        
    Returns:
        List of sensor names in order
    """
    # Define the namespace
    ns = {'mvn': 'http://www.xsens.com/mvn/mvnx'}
    
    # Find sensors using the correct namespace
    sensors = root.findall(".//mvn:sensor", ns)
    return [sensor.get('label', f'sensor_{i}') for i, sensor in enumerate(sensors)]

def parse_mvnx(file_path: str) -> pd.DataFrame:
    """
    Parse MVNX file and extract sensor data into a pandas DataFrame.
    """
    # Define the namespace
    ns = {'mvn': 'http://www.xsens.com/mvn/mvnx'}
    
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Get subject info
    subject = root.find('.//mvn:subject', ns)
    if subject is not None:
        print(f"Subject: {subject.get('label')}")
        print(f"Frame rate: {subject.get('frameRate')}")
    
    # Find all frames
    frames = root.findall(".//mvn:frame", ns)
    print(f"Found {len(frames)} frames")
    
    # Initialize data storage
    data: Dict[str, List[float]] = {
        'time': [],
        'frame_index': [],
        'frame_type': []
    }
    
    # Add columns for each data type
    data_types = ['orientation', 'position', 'velocity', 'acceleration', 'angularVelocity']
    
    # Initialize all possible columns by scanning all frames for maximum values
    max_values = {}
    for data_type in data_types:
        max_values[data_type] = 0
        for frame in frames:
            element = frame.find(f'mvn:{data_type}', ns)
            if element is not None and element.text:
                values = element.text.strip().split()
                max_values[data_type] = max(max_values[data_type], len(values))
    
    # Create columns based on maximum values found
    for data_type in data_types:
        for i in range(max_values[data_type]):
            data[f'{data_type}_{i}'] = []
    
    print("Data columns created:", len(data.keys()))
    
    # Extract data from each frame
    for i, frame in enumerate(frames):
        # Get frame attributes, with safe defaults
        time_str = frame.get('time', '')
        index_str = frame.get('index', '')
        
        try:
            time = float(time_str) if time_str else float(i)
            frame_index = int(index_str) if index_str else i
        except ValueError:
            time = float(i)
            frame_index = i
            
        data['time'].append(time)
        data['frame_index'].append(frame_index)
        data['frame_type'].append(frame.get('type', 'normal'))
        
        # Get data for each type
        for data_type in data_types:
            element = frame.find(f'mvn:{data_type}', ns)
            values = element.text.strip().split() if element is not None and element.text else []
            
            # Fill in values or zeros
            for j in range(max_values[data_type]):
                value = float(values[j]) if j < len(values) else 0.0
                data[f'{data_type}_{j}'].append(value)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Set time as index
    df.set_index('time', inplace=True)
    
    return df

def main():
    input_path = Path("data/Run 2 - Frontside 10 Stalefish.mvnx")
    output_path = input_path.with_suffix('.csv')
    
    try:
        print(f"Processing file: {input_path}")
        df = parse_mvnx(str(input_path))
        df.to_csv(output_path)
        print(f"\nSuccessfully converted {input_path} to {output_path}")
        print(f"DataFrame shape: {df.shape}")
        print("\nColumns:", ', '.join(df.columns[:6]))
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
