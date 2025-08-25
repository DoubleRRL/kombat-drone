"""
Professional GUI for Combat Video Detection System
User-friendly interface for demonstrating CV capabilities to departments
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import time
import cv2
from pathlib import Path
from PIL import Image, ImageTk
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent))

from custom_video_processor import CustomVideoProcessor


class VideoProcessorGUI:
    """
    Professional GUI for video processing demonstrations
    Perfect for showing off to department heads and external stakeholders
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Combat CV Detection System - Professional Demo")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Processing state
        self.processor = None
        self.processing_thread = None
        self.is_processing = False
        self.message_queue = queue.Queue()
        
        # Video preview
        self.preview_cap = None
        self.preview_running = False
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        self.setup_layout()
        
        # Start message processing
        self.root.after(100, self.process_messages)
    
    def setup_styles(self):
        """Setup professional styling"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors for professional look
        self.style.configure('Title.TLabel', 
                           font=('Arial', 16, 'bold'),
                           foreground='#00ff00',
                           background='#2b2b2b')
        
        self.style.configure('Header.TLabel',
                           font=('Arial', 12, 'bold'),
                           foreground='#ffffff',
                           background='#2b2b2b')
        
        self.style.configure('Info.TLabel',
                           font=('Arial', 10),
                           foreground='#cccccc',
                           background='#2b2b2b')
        
        self.style.configure('Process.TButton',
                           font=('Arial', 12, 'bold'),
                           foreground='#ffffff')
        
        self.style.configure('Stop.TButton',
                           font=('Arial', 12, 'bold'),
                           foreground='#ff0000')
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Main title
        self.title_label = ttk.Label(
            self.root,
            text="🎯 COMBAT CV DETECTION SYSTEM",
            style='Title.TLabel'
        )
        
        # Input section
        self.input_frame = ttk.LabelFrame(self.root, text="Input Configuration", padding=10)
        
        # Video file selection
        ttk.Label(self.input_frame, text="Video File:", style='Header.TLabel').grid(row=0, column=0, sticky='w', pady=5)
        
        self.video_path_var = tk.StringVar()
        self.video_entry = ttk.Entry(self.input_frame, textvariable=self.video_path_var, width=50)
        self.video_entry.grid(row=0, column=1, padx=10, pady=5)
        
        self.browse_button = ttk.Button(
            self.input_frame,
            text="Browse",
            command=self.browse_video_file
        )
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Output file selection
        ttk.Label(self.input_frame, text="Output File:", style='Header.TLabel').grid(row=1, column=0, sticky='w', pady=5)
        
        self.output_path_var = tk.StringVar()
        self.output_entry = ttk.Entry(self.input_frame, textvariable=self.output_path_var, width=50)
        self.output_entry.grid(row=1, column=1, padx=10, pady=5)
        
        self.output_browse_button = ttk.Button(
            self.input_frame,
            text="Browse",
            command=self.browse_output_file
        )
        self.output_browse_button.grid(row=1, column=2, padx=5, pady=5)
        
        # Detection settings
        self.settings_frame = ttk.LabelFrame(self.root, text="Detection Settings", padding=10)
        
        # Detection method
        ttk.Label(self.settings_frame, text="Detection Method:", style='Header.TLabel').grid(row=0, column=0, sticky='w', pady=5)
        
        self.method_var = tk.StringVar(value="thermal")
        method_combo = ttk.Combobox(
            self.settings_frame,
            textvariable=self.method_var,
            values=["thermal", "lightweight", "full"],
            state="readonly",
            width=15
        )
        method_combo.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        
        # Method descriptions
        method_info = {
            "thermal": "Fast temperature-based detection",
            "lightweight": "YOLO-based detection (balanced)",
            "full": "Complete multimodal pipeline (comprehensive)"
        }
        
        self.method_info_var = tk.StringVar(value=method_info["thermal"])
        ttk.Label(self.settings_frame, textvariable=self.method_info_var, style='Info.TLabel').grid(row=0, column=2, padx=10, pady=5, sticky='w')
        
        def update_method_info(*args):
            self.method_info_var.set(method_info.get(self.method_var.get(), ""))
        
        self.method_var.trace('w', update_method_info)
        
        # Confidence threshold
        ttk.Label(self.settings_frame, text="Confidence Threshold:", style='Header.TLabel').grid(row=1, column=0, sticky='w', pady=5)
        
        self.confidence_var = tk.DoubleVar(value=0.3)
        confidence_scale = ttk.Scale(
            self.settings_frame,
            from_=0.1,
            to=0.9,
            variable=self.confidence_var,
            orient='horizontal',
            length=200
        )
        confidence_scale.grid(row=1, column=1, padx=10, pady=5, sticky='w')
        
        self.confidence_label = ttk.Label(self.settings_frame, text="0.30", style='Info.TLabel')
        self.confidence_label.grid(row=1, column=2, padx=10, pady=5, sticky='w')
        
        def update_confidence(*args):
            self.confidence_label.config(text=f"{self.confidence_var.get():.2f}")
        
        self.confidence_var.trace('w', update_confidence)
        
        # Max frames (for testing)
        ttk.Label(self.settings_frame, text="Max Frames (0 = all):", style='Header.TLabel').grid(row=2, column=0, sticky='w', pady=5)
        
        self.max_frames_var = tk.IntVar(value=0)
        max_frames_spin = ttk.Spinbox(
            self.settings_frame,
            from_=0,
            to=10000,
            textvariable=self.max_frames_var,
            width=10
        )
        max_frames_spin.grid(row=2, column=1, padx=10, pady=5, sticky='w')
        
        ttk.Label(self.settings_frame, text="(useful for testing large videos)", style='Info.TLabel').grid(row=2, column=2, padx=10, pady=5, sticky='w')
        
        # Control buttons
        self.control_frame = ttk.Frame(self.root)
        
        self.process_button = ttk.Button(
            self.control_frame,
            text="🚀 START PROCESSING",
            command=self.start_processing,
            style='Process.TButton'
        )
        self.process_button.pack(side='left', padx=10, pady=10)
        
        self.stop_button = ttk.Button(
            self.control_frame,
            text="⏹ STOP",
            command=self.stop_processing,
            state='disabled',
            style='Stop.TButton'
        )
        self.stop_button.pack(side='left', padx=10, pady=10)
        
        self.preview_button = ttk.Button(
            self.control_frame,
            text="👁 PREVIEW VIDEO",
            command=self.preview_video
        )
        self.preview_button.pack(side='left', padx=10, pady=10)
        
        # Export options
        self.export_button = ttk.Button(
            self.control_frame,
            text="📁 OPEN OUTPUT FOLDER",
            command=self.open_output_folder,
            state='disabled'
        )
        self.export_button.pack(side='left', padx=10, pady=10)
        
        # Progress section with live banner
        self.progress_frame = ttk.LabelFrame(self.root, text="Processing Status", padding=10)
        
        # Live progress banner
        self.banner_frame = tk.Frame(self.progress_frame, bg='#1a1a1a', height=60)
        self.banner_frame.pack(fill='x', pady=5)
        self.banner_frame.pack_propagate(False)
        
        self.banner_text = tk.Label(
            self.banner_frame,
            text="🎯 Ready to process video...",
            bg='#1a1a1a',
            fg='#00ff00',
            font=('Arial', 14, 'bold'),
            anchor='center'
        )
        self.banner_text.pack(expand=True, fill='both')
        
        # Detailed progress info
        self.progress_var = tk.StringVar(value="Select a video file to begin processing")
        self.progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_var, style='Info.TLabel')
        self.progress_label.pack(anchor='w', pady=5)
        
        # Progress bar with percentage
        self.progress_container = tk.Frame(self.progress_frame)
        self.progress_container.pack(fill='x', pady=5)
        
        self.progress_bar = ttk.Progressbar(
            self.progress_container,
            mode='determinate',
            length=350
        )
        self.progress_bar.pack(side='left', fill='x', expand=True)
        
        self.progress_percent = tk.Label(
            self.progress_container,
            text="0%",
            bg='#2b2b2b',
            fg='#ffffff',
            font=('Arial', 10, 'bold'),
            width=5
        )
        self.progress_percent.pack(side='right', padx=(10, 0))
        
        # Statistics display
        self.stats_frame = ttk.LabelFrame(self.progress_frame, text="Real-time Statistics", padding=5)
        
        self.stats_text = scrolledtext.ScrolledText(
            self.stats_frame,
            height=8,
            width=60,
            bg='#1a1a1a',
            fg='#00ff00',
            font=('Consolas', 10)
        )
        self.stats_text.pack(fill='both', expand=True)
        
        # Log output
        self.log_frame = ttk.LabelFrame(self.root, text="Processing Log", padding=10)
        
        self.log_text = scrolledtext.ScrolledText(
            self.log_frame,
            height=12,
            bg='#1a1a1a',
            fg='#ffffff',
            font=('Consolas', 9)
        )
        self.log_text.pack(fill='both', expand=True)
        
        # Add initial welcome message
        self.log_message("🎯 Combat CV Detection System Ready")
        self.log_message("   Select a video file and configure detection settings")
        self.log_message("   Perfect for demonstrating capabilities to departments\n")
    
    def setup_layout(self):
        """Setup widget layout"""
        self.title_label.pack(pady=20)
        
        self.input_frame.pack(fill='x', padx=20, pady=10)
        self.settings_frame.pack(fill='x', padx=20, pady=10)
        self.control_frame.pack(pady=20)
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.root)
        
        # Progress tab
        progress_tab = ttk.Frame(self.notebook)
        self.notebook.add(progress_tab, text="Processing Status")
        
        self.progress_frame.pack(in_=progress_tab, fill='both', expand=True, padx=20, pady=10)
        self.stats_frame.pack(in_=self.progress_frame, fill='both', expand=True, pady=10)
        
        # Log tab
        log_tab = ttk.Frame(self.notebook)
        self.notebook.add(log_tab, text="System Log")
        
        self.log_frame.pack(in_=log_tab, fill='both', expand=True, padx=20, pady=10)
        
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)
    
    def browse_video_file(self):
        """Browse for input video file"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        
        if filename:
            self.video_path_var.set(filename)
            
            # Auto-generate output filename
            input_path = Path(filename)
            output_path = input_path.parent / f"{input_path.stem}_detected.mp4"
            self.output_path_var.set(str(output_path))
            
            self.log_message(f"Selected video: {filename}")
    
    def browse_output_file(self):
        """Browse for output video file"""
        filename = filedialog.asksaveasfilename(
            title="Save Processed Video As",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.output_path_var.set(filename)
    
    def preview_video(self):
        """Preview selected video"""
        video_path = self.video_path_var.get()
        if not video_path or not Path(video_path).exists():
            messagebox.showerror("Error", "Please select a valid video file first")
            return
        
        self.log_message(f"Opening video preview: {Path(video_path).name}")
        
        # Simple preview using cv2
        def show_preview():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                messagebox.showerror("Error", "Cannot open video file")
                return
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize for preview
                height, width = frame.shape[:2]
                if width > 800:
                    scale = 800 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                cv2.imshow("Video Preview (Press 'q' to close)", frame)
                
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        # Run preview in separate thread
        threading.Thread(target=show_preview, daemon=True).start()
    
    def start_processing(self):
        """Start video processing"""
        # Validate inputs
        video_path = self.video_path_var.get()
        output_path = self.output_path_var.get()
        
        if not video_path:
            messagebox.showerror("Error", "Please select an input video file")
            return
        
        if not Path(video_path).exists():
            messagebox.showerror("Error", "Input video file does not exist")
            return
        
        if not output_path:
            messagebox.showerror("Error", "Please specify an output file path")
            return
        
        # Update UI state
        self.is_processing = True
        self.process_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.export_button.config(state='disabled')
        
        # Reset progress
        self.progress_bar['value'] = 0
        self.progress_percent.config(text="0%")
        self.update_banner("🚀 Initializing detection pipeline...", '#ff6600')
        
        self.log_message("🚀 Starting video processing...")
        self.log_message(f"   Input: {video_path}")
        self.log_message(f"   Output: {output_path}")
        self.log_message(f"   Method: {self.method_var.get()}")
        self.log_message(f"   Confidence: {self.confidence_var.get():.2f}")
        
        # Start processing in separate thread
        self.processing_thread = threading.Thread(
            target=self.process_video_thread,
            daemon=True
        )
        self.processing_thread.start()
    
    def process_video_thread(self):
        """Video processing thread"""
        try:
            # Create processor
            max_frames = self.max_frames_var.get() if self.max_frames_var.get() > 0 else None
            
            processor = CustomVideoProcessor(
                input_video_path=self.video_path_var.get(),
                output_video_path=self.output_path_var.get(),
                detection_method=self.method_var.get(),
                confidence_threshold=self.confidence_var.get(),
                show_fps=True
            )
            
            # Initialize and get video info
            processor.initialize()
            
            # Send video info to GUI
            self.message_queue.put({
                'type': 'video_info',
                'fps': processor.fps,
                'size': (processor.frame_width, processor.frame_height),
                'total_frames': processor.total_frames
            })
            
            # Process with custom progress callback
            self.process_with_progress(processor, max_frames)
            
            self.message_queue.put({
                'type': 'complete',
                'output_path': self.output_path_var.get()
            })
            
        except Exception as e:
            self.message_queue.put({
                'type': 'error',
                'message': str(e)
            })
    
    def process_with_progress(self, processor, max_frames):
        """Process video with progress updates"""
        if not processor.cap:
            processor.initialize()
        
        frame_count = 0
        processing_times = []
        detection_counts = []
        start_time = time.time()
        
        while self.is_processing:
            ret, frame = processor.cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            # Process frame
            frame_start = time.time()
            processed_frame = processor._process_single_frame(frame, frame_count)
            processing_time = (time.time() - frame_start) * 1000
            
            processing_times.append(processing_time)
            
            # Write processed frame
            processor.writer.write(processed_frame)
            
            frame_count += 1
            
            # Send progress update every 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / processor.total_frames) * 100
                avg_time = sum(processing_times[-30:]) / min(30, len(processing_times))
                current_fps = 1000 / avg_time if avg_time > 0 else 0
                
                elapsed_time = time.time() - start_time
                
                self.message_queue.put({
                    'type': 'progress',
                    'frame': frame_count,
                    'total': processor.total_frames,
                    'progress': progress,
                    'fps': current_fps,
                    'avg_time': avg_time,
                    'elapsed': elapsed_time
                })
        
        processor.cleanup()
    
    def stop_processing(self):
        """Stop video processing"""
        self.is_processing = False
        self.log_message("⏹ Stopping processing...")
        
        # Update UI state
        self.process_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        self.progress_var.set("Processing stopped by user")
        self.update_banner("⏹ Processing stopped", '#ffaa00')
    
    def process_messages(self):
        """Process messages from worker thread"""
        try:
            while True:
                message = self.message_queue.get_nowait()
                
                if message['type'] == 'video_info':
                    info_text = f"Video Info:\n"
                    info_text += f"  Resolution: {message['size'][0]}x{message['size'][1]}\n"
                    info_text += f"  FPS: {message['fps']:.1f}\n"
                    info_text += f"  Total Frames: {message['total_frames']}\n"
                    info_text += f"  Duration: {message['total_frames']/message['fps']:.1f}s\n\n"
                    
                    self.stats_text.insert(tk.END, info_text)
                    self.stats_text.see(tk.END)
                
                elif message['type'] == 'progress':
                    progress_text = f"Frame {message['frame']}/{message['total']} "
                    progress_text += f"({message['progress']:.1f}%)\n"
                    progress_text += f"Processing: {message['avg_time']:.1f}ms/frame\n"
                    progress_text += f"Speed: {message['fps']:.1f} fps\n"
                    progress_text += f"Elapsed: {message['elapsed']:.1f}s\n\n"
                    
                    # Update stats display
                    self.stats_text.insert(tk.END, progress_text)
                    self.stats_text.see(tk.END)
                    
                    # Update progress components
                    progress = message['progress']
                    self.progress_bar['value'] = progress
                    self.progress_percent.config(text=f"{progress:.0f}%")
                    
                    # Update banner with dynamic messages based on progress
                    if progress < 25:
                        self.update_banner("🔄 Processing video... warming up", '#00aaff')
                    elif progress < 50:
                        self.update_banner("⚡ Processing video... hitting stride", '#00ff88')
                    elif progress < 75:
                        self.update_banner("🔥 Processing video... cruising along", '#88ff00')
                    elif progress < 95:
                        self.update_banner("🚀 Processing video... almost there!", '#ffaa00')
                    else:
                        self.update_banner("✨ Processing video... finishing up!", '#ff6600')
                    
                    # Update progress label
                    self.progress_var.set(f"Processing frame {message['frame']}/{message['total']} ({message['progress']:.1f}%)")
                
                elif message['type'] == 'complete':
                    self.log_message("✅ Video processing complete!")
                    self.log_message(f"   Output saved: {message['output_path']}")
                    
                    # Final progress update
                    self.progress_bar['value'] = 100
                    self.progress_percent.config(text="100%")
                    self.progress_var.set("Processing complete! Video ready for export.")
                    
                    # Celebration banner
                    self.update_banner("🎉 Export complete! Video ready to view", '#00ff00')
                    
                    self.process_button.config(state='normal')
                    self.stop_button.config(state='disabled')
                    self.export_button.config(state='normal')
                    self.is_processing = False
                    
                    # Success notification with export options
                    result = messagebox.askyesno(
                        "Export Complete! 🎉", 
                        f"Video processing complete with detections!\n\n"
                        f"Output saved to:\n{message['output_path']}\n\n"
                        f"Would you like to open the output folder?"
                    )
                    
                    if result:
                        self.open_output_folder()
                
                elif message['type'] == 'error':
                    self.log_message(f"❌ Error: {message['message']}")
                    
                    self.progress_var.set("Error occurred during processing")
                    self.update_banner("❌ Processing failed", '#ff0000')
                    
                    self.process_button.config(state='normal')
                    self.stop_button.config(state='disabled')
                    self.is_processing = False
                    
                    messagebox.showerror("Error", f"Processing failed:\n{message['message']}")
        
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_messages)
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
    
    def update_banner(self, text, color='#00ff00'):
        """Update the live progress banner with cute animations"""
        self.banner_text.config(text=text, fg=color)
        
        # Subtle pulse animation for active states
        if 'Processing' in text or 'Initializing' in text:
            self.root.after(500, lambda: self.pulse_banner(color))
    
    def pulse_banner(self, original_color):
        """Create a subtle pulsing effect for the banner"""
        if self.is_processing and 'Processing' in self.banner_text.cget('text'):
            # Pulse to lighter color
            lighter_color = '#ffffff' if original_color == '#00ff00' else original_color
            self.banner_text.config(fg=lighter_color)
            
            # Pulse back after 300ms
            self.root.after(300, lambda: self.banner_text.config(fg=original_color))
            
            # Schedule next pulse
            self.root.after(1000, lambda: self.pulse_banner(original_color))
    
    def open_output_folder(self):
        """Open the output folder in file manager"""
        output_path = self.output_path_var.get()
        if not output_path:
            messagebox.showwarning("No Output", "No output file specified")
            return
        
        output_dir = Path(output_path).parent
        
        try:
            import subprocess
            import platform
            
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(["open", str(output_dir)])
            elif system == "Windows":
                subprocess.run(["explorer", str(output_dir)])
            else:  # Linux
                subprocess.run(["xdg-open", str(output_dir)])
                
            self.log_message(f"📁 Opened output folder: {output_dir}")
            
        except Exception as e:
            self.log_message(f"❌ Could not open folder: {e}")
            messagebox.showerror("Error", f"Could not open output folder:\n{e}")


def main():
    """Launch the GUI application"""
    root = tk.Tk()
    app = VideoProcessorGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.quit()


if __name__ == "__main__":
    main()
