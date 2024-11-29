import tkinter as tk

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paint Application")
        self.root.geometry("800x600")
        
        # Default color and brush size
        self.brush_color = "black"
        self.brush_size = 5
        self.is_eraser = False  # Flag to track if the eraser is active
        
        # Create the canvas
        self.canvas = tk.Canvas(self.root, bg="white", width=800, height=500)
        self.canvas.pack(pady=20)
        
        # Tools frame
        self.tools_frame = tk.Frame(self.root)
        self.tools_frame.pack()
        
        # Add color palette
        self.create_color_palette(self.tools_frame)
        
        # Add brush size slider
        self.brush_size_slider = tk.Scale(self.tools_frame, from_=1, to=20, orient=tk.HORIZONTAL, label="Brush Size")
        self.brush_size_slider.set(self.brush_size)
        self.brush_size_slider.grid(row=0, column=len(self.color_buttons), padx=5)
        
        # Add clear button
        clear_btn = tk.Button(self.tools_frame, text="Clear Canvas", command=self.clear_canvas)
        clear_btn.grid(row=0, column=len(self.color_buttons) + 1, padx=5)
        
        # Add eraser button
        erase_btn = tk.Button(self.tools_frame, text="Eraser", command=self.use_eraser)
        erase_btn.grid(row=0, column=len(self.color_buttons) + 2, padx=5)
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        self.last_x, self.last_y = None, None  # To track the last position for smooth drawing
        self.drawing_objects = []  # List to store IDs of drawn objects
    
    def create_color_palette(self, parent):
        """Create a set of predefined color buttons."""
        self.color_buttons = []
        colors = ["black", "red", "green", "blue", "yellow", "orange", "purple", "brown"]
        for idx, color in enumerate(colors):
            btn = tk.Button(parent, bg=color, width=3, height=1, command=lambda c=color: self.set_brush_color(c))
            btn.grid(row=0, column=idx, padx=2)
            self.color_buttons.append(btn)
    
    def set_brush_color(self, color):
        """Set the brush color to the selected color."""
        if not self.is_eraser:
            self.brush_color = color
    
    def clear_canvas(self):
        """Clear the entire canvas."""
        self.canvas.delete("all")
        self.drawing_objects.clear()  # Clear the list of drawing objects
    
    def use_eraser(self):
        """Switch to eraser mode, where the brush deletes painted areas."""
        self.is_eraser = True
        self.brush_color = "white"
    
    def paint(self, event):
        """Draw or erase on the canvas."""
        brush_size = self.brush_size_slider.get()
        if self.last_x and self.last_y:
            if self.is_eraser:
                # Erase functionality: Remove any object that intersects with the brush
                self.erase(event.x, event.y, brush_size)
            else:
                # Draw on canvas
                line_id = self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                                  width=brush_size, fill=self.brush_color, capstyle=tk.ROUND, smooth=True)
                self.drawing_objects.append(line_id)  # Store the object ID
            
        self.last_x, self.last_y = event.x, event.y
    
    def erase(self, x, y, size):
        """Erase the area where the brush moves."""
        # Delete lines within the area of the eraser
        for obj_id in self.drawing_objects:
            coords = self.canvas.coords(obj_id)
            if self.is_within_eraser_area(coords, x, y, size):
                self.canvas.delete(obj_id)
                self.drawing_objects.remove(obj_id)
    
    def is_within_eraser_area(self, coords, x, y, size):
        """Check if the drawn line is within the area that should be erased."""
        x1, y1, x2, y2 = coords
        return abs(x - (x1 + x2) / 2) < size and abs(y - (y1 + y2) / 2) < size
    
    def reset(self, event):
        """Reset the last position."""
        self.last_x, self.last_y = None, None

# Create the application
if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()
