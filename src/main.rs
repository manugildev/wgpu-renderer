use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use futures::executor::block_on;

struct State {
    surface: wgpu::Surface,
    // Logical Device
    device: wgpu::Device,
    queue: wgpu::Queue,
    swap_chain_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    size: winit::dpi::PhysicalSize<u32>, // INFO: PhysicalSize takes into account device's scale factor
}

impl State {
    async fn new(window: &Window) -> Self {
        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };

        // Create Adapter
        let adapter_options = &wgpu::RequestAdapterOptions {
            // Default gets LowP on battery and HighP when on mains
            power_preference: wgpu::PowerPreference::Default,
            compatible_surface: Some(&surface),
        };
        // The adapter identifies both an instance of a physical hardware accelerator (CPU, GPU),
        // and an instance of a browser's implementation of WebGPU on top of the accelerator
        let adapter = instance.request_adapter(adapter_options).await.unwrap();

        // Create Device and Queue
        let desc = &wgpu::DeviceDescriptor {
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
            shader_validation: true,
        };
        let (device, queue) = adapter.request_device(desc, None).await.unwrap();

        // Create SwapChain
        let size = window.inner_size(); // INFO: Has into account the scale factor
        let swap_chain_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb, // TODO: Should be swap_chain_get_current_texture_view but not available atm
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        let swap_chain = device.create_swap_chain(&surface, &swap_chain_desc);

        return State {
            surface,
            device,
            queue,
            swap_chain_desc,
            swap_chain,
            size,
        };
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.swap_chain_desc.width = new_size.width;
        self.swap_chain_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.swap_chain_desc);
    }

    // Returns a bool to indicate whether an event has been fully processed. If `true` the main
    // loop won't process the event any further
    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {}

    fn render(&mut self) -> Result<(), wgpu::SwapChainError> {
        // Get next frame
        let frame = self.swap_chain.get_current_frame()?.output;

        // Create command encoder
        let command_encoder_desc = wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        };
        let mut encoder = self.device.create_command_encoder(&command_encoder_desc);

        {
            // Create Render Pass
            let clear_color = wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0, };
            let render_pass_desc = wgpu::RenderPassDescriptor {
                // Color Attachments
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view, // Current frame texture view
                    resolve_target: None,    // Only used if multisampling is enabled
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: true,
                    },
                }],
                // Depth Stencil Attachments
                depth_stencil_attachment: None,
            };
            /*let _render_pass =*/ encoder.begin_render_pass(&render_pass_desc);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        return Ok(());
    }
}

fn handle_keyboard_input(state: &mut State, input: KeyboardInput, control_flow: &mut ControlFlow) {
    if input.state == ElementState::Pressed {
        match input.virtual_keycode {
            Some(VirtualKeyCode::Escape) => {
                *control_flow = ControlFlow::Exit;
            }
            _ => {}
        }
    }
}

fn handle_window_events(state: &mut State, event: WindowEvent, control_flow: &mut ControlFlow) {
    match event {
        WindowEvent::KeyboardInput { input, ../*device_id, is_synthetic*/ } => {
            handle_keyboard_input(state, input, control_flow);
        },
        WindowEvent::Resized(physical_size) => {
            state.resize(physical_size)
        },
        WindowEvent::ScaleFactorChanged {new_inner_size, ../*scale_factor*/ } => {
            state.resize(*new_inner_size)
        },
        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
        _ => {}
    }
}

fn handle_redraw_requested(state: &mut State, control_flow: &mut ControlFlow) {
    state.update();
    match state.render() {
        Err(wgpu::SwapChainError::Lost) => state.resize(state.size),
        Err(wgpu::SwapChainError::OutOfMemory) => *control_flow = ControlFlow::Exit,
        Err(e) => eprintln!("{:?}", e),
        Ok(_) => {}
    }
}

fn main() {
    env_logger::init(); // INFO: error!, warn!, info!, debug! and trace!
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    let mut state = block_on(State::new(&window));

    // INFO: move -> moves any variables you reference which are outside the scope of the closure into the closure's object.
    event_loop.run(move |event, _event_loop_window_target, control_flow| {
        match event {
            Event::WindowEvent { event, window_id } => {
                if window_id == window.id() && !state.input(&event) {
                    handle_window_events(&mut state, event, control_flow);
                }
            }
            Event::RedrawRequested(_window_id) => {
                handle_redraw_requested(&mut state, control_flow);
            }
            Event::MainEventsCleared => {
                // Emitted when all of the event loop's input events have been processed and redraw processing
                // is about to begin.
                window.request_redraw();
            }
            _ => {}
        };
    });
}
