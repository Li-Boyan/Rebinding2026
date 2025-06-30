import numpy as np
import matplotlib.pyplot as plt
from nd2reader import ND2Reader
import ipywidgets as widgets
from IPython.display import display

def visualize_nd2(file_path):
    """
    Interactive visualization of ND2 file with multiple positions and gamma adjustment
    
    Parameters:
    -----------
    file_path : str
        Path to the ND2 file
    """
    # Create a figure first - outside the callback
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # This function contains all the visualization logic
    # but keeps the ND2 file open throughout
    def setup_visualization():
        # Open the ND2 file - this needs to stay open
        images = ND2Reader(file_path)
        images.iter_axes = ""
        
        # Get dimensions
        positions = images.sizes.get('v', 1)  # Get number of positions, default 1
        channels = images.sizes.get('c', 1)   # Get number of channels, default 1
        z_slices = images.sizes.get('z', 1)   # Get number of z-slices, default 1
        time_points = images.sizes.get('t', 1)  # Get number of time points, default 1
        
        # Create widgets for controlling the view
        position_slider = widgets.IntSlider(
            value=0, min=0, max=max(0, positions-1), 
            description='Position:', disabled=(positions <= 1)
        )
        
        channel_slider = widgets.IntSlider(
            value=0, min=0, max=max(0, channels-1), 
            description='Channel:', disabled=(channels <= 1)
        )
        
        z_slider = widgets.IntSlider(
            value=0, min=0, max=max(0, z_slices-1), 
            description='Z-slice:', disabled=(z_slices <= 1)
        )
        
        time_slider = widgets.IntSlider(
            value=0, min=0, max=max(0, time_points-1), 
            description='Time:', disabled=(time_points <= 1)
        )
        
        # Change from brightness to gamma adjustment
        gamma_slider = widgets.FloatSlider(
            value=1.0, min=0.1, max=3.0, step=0.1,
            description='Gamma:'
        )
        
        # Create a function that updates the plot
        def update_plot(change):
            # Get current values from all sliders
            position = position_slider.value
            channel = channel_slider.value
            z = z_slider.value
            time_point = time_slider.value
            gamma = gamma_slider.value
            
            # Set the desired indices
            if positions > 1:
                images.default_coords['v'] = position
            if channels > 1:
                images.default_coords['c'] = channel
            if z_slices > 1:
                images.default_coords['z'] = z
            if time_points > 1:
                images.default_coords['t'] = time_point

            print(images.default_coords)
            
            # Get the image
            frame = images[0].astype(float)
            
            # Normalize to 0-1 range for gamma correction
            if frame.max() > 0:  # Avoid division by zero
                frame = frame / frame.max()
            
            # Apply gamma correction: I_out = I_in^(1/gamma)
            frame = np.power(frame, 1.0/gamma)
            
            # Clear the previous plot and create a new one
            ax.clear()
            ax.imshow(frame, cmap='gray')
            
            # Set title with metadata
            title = f"Position: {position}/{positions-1}"
            if channels > 1:
                title += f", Channel: {channel}/{channels-1}"
            if z_slices > 1:
                title += f", Z: {z}/{z_slices-1}"
            if time_points > 1:
                title += f", Time: {time_point}/{time_points-1}"
            ax.set_title(title)
            
            # Update the plot
            fig.canvas.draw_idle()
        
        # Register the update function with all sliders
        position_slider.observe(update_plot, names='value')
        channel_slider.observe(update_plot, names='value')
        z_slider.observe(update_plot, names='value')
        time_slider.observe(update_plot, names='value')
        gamma_slider.observe(update_plot, names='value')
        
        # Initial plot
        update_plot(None)
        
        # Create a widget container for the sliders
        controls = widgets.VBox([
            position_slider, 
            channel_slider, 
            z_slider, 
            time_slider,
            gamma_slider
        ])
        
        # Display the widgets
        display(controls)
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        # Print metadata
        print(f"File metadata:")
        print(f"  • Dimensions: {images.sizes}")
        print(f"  • Pixel size: {images.metadata.get('pixel_microns', 'Not available')}")
        
        # Return the ND2Reader object to prevent it from being closed
        return images
    
    # Call the setup function and keep a reference to the returned ND2Reader
    nd2_file = setup_visualization()
    
    # Return the file reference to prevent premature garbage collection
    return nd2_file
    
# Usage example:
# visualize_nd2('your_file.nd2')