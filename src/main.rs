use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use futures::executor::block_on;
use wgpu::util::DeviceExt;


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}
impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        let attrib_descs = &[
            // Position
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float3,
            },
            // Color
            wgpu::VertexAttributeDescriptor {
                offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                shader_location: 1,
                format: wgpu::VertexFormat::Float3,
            }
        ];

        return wgpu::VertexBufferDescriptor {
            stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: attrib_descs
            //attributes: &wgpu::vertex_attr_array![0 => Float3, 1 => Float3],
        };
    }
}

const VERTICES: &[Vertex] = &[
    Vertex { position: [-0.08,  0.49, 0.0], color: [1.0, 0.0, 0.0] }, // A
    Vertex { position: [-0.49,  0.06, 0.0], color: [0.0, 1.0, 1.0] }, // B
    Vertex { position: [-0.21, -0.44, 0.0], color: [0.0, 1.0, 0.0] }, // C
    Vertex { position: [ 0.35, -0.34, 0.0], color: [1.0, 0.0, 1.0] }, // D
    Vertex { position: [ 0.44,  0.23, 0.0], color: [0.0, 0.0, 1.0] }, // E
];

const INDICES: &[u16] = &[
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
];

struct State {
    surface: wgpu::Surface,
    // Logical Device
    device: wgpu::Device,
    queue: wgpu::Queue,
    swap_chain_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    size: winit::dpi::PhysicalSize<u32>, // INFO: PhysicalSize takes into account device's scale factor
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
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
            power_preference: wgpu::PowerPreference::default(),
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

        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
        let diffuse_rgba = diffuse_image.as_rgba8().unwrap();

        use image::GenericImageView;
        let dimensions = diffuse_image.dimensions();

        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth: 1,
        };

        let diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // SAMPLED tells wgpu that we want to use this texture in shaders
            // COPY_DST means that we want to copy data to this texture
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            label: Some("diffuse_texture"),
        });

        queue.write_texture(
            wgpu::TextureCopyView {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            diffuse_rgba,
            wgpu::TextureDataLayout {
                offset:0,
                bytes_per_row: 4 * dimensions.0,
                rows_per_image: dimensions.1,
            },
            texture_size,
        );

        let diffuse_texture_view = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create ShaderModules
        let vs_module = device.create_shader_module(wgpu::include_spirv!("../shaders/shader.vert.spv"));
        let fs_module = device.create_shader_module(wgpu::include_spirv!("../shaders/shader.frag.spv"));

        // Create Vertex Buffer
        let buffer_init_desc = wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES), // INFO: cast to &[u8]
            usage: wgpu::BufferUsage::VERTEX,
        };

        let vertex_buffer = device.create_buffer_init(&buffer_init_desc);
        let num_vertices = VERTICES.len() as u32;

        // Create Vertex Buffer
        let buffer_init_desc = wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES), // INFO: cast to &[u8]
            usage: wgpu::BufferUsage::INDEX,
        };

        let index_buffer = device.create_buffer_init(&buffer_init_desc);
        let num_indices = INDICES.len() as u32;

        // Create Pipeline Layout
        let pipeline_layout_desc = wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        };
        let render_pipeline_layout = device.create_pipeline_layout(&pipeline_layout_desc);

        // Create Render Pipeline
        let render_pipeline_desc = wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor{
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                clamp_depth: false,
            }),
            color_states: &[wgpu::ColorStateDescriptor{ // Define how colors are stored and processed
                format: swap_chain_desc.format,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[Vertex::desc()],
            },
            sample_count: 1,
            sample_mask: !0, // Use all samples
            alpha_to_coverage_enabled: false,
        };
        let render_pipeline = device.create_render_pipeline(&render_pipeline_desc);

        return State {
            surface,
            device,
            queue,
            swap_chain_desc,
            swap_chain,
            size,
            render_pipeline,
            vertex_buffer,
            num_vertices,
            index_buffer,
            num_indices,
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
            let clear_color = wgpu::Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0, };
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
            let mut render_pass = encoder.begin_render_pass(&render_pass_desc);
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..));
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
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

    let window = WindowBuilder::new().with_title("WGPU Renderer").build(&event_loop).unwrap();

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
