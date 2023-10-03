pub mod vert_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
        #version 460

        layout(location = 0) in vec2 position;

        void main()
        {
            gl_Position = vec4(position, 0.0, 1.0);
        }
        "
    }
}

pub mod frag_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
        #version 460
        
        layout(location = 0) out vec4 f_color;

        void main()
        {
            f_color = vec4(1.0, 0.0, 0.0, 1.0);
        }
        "
    }
}

pub mod vertex_shader_for_moving {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
        #version 460

        layout(location = 0) in vec2 position;

        layout(set = 0, binding = 0) uniform Data {
            vec3 color;
            vec2 position;
        } uniforms;

        layout(location = 0) out vec3 outColor;

        void main() {
            outColor = uniforms.color;
            gl_Position = vec4(
                position.x + uniforms.position.x, 
                position.y + uniforms.position.y, 
                0.0, 
                1.0
            );
        }
        "
    }
}

pub mod fragment_shader_for_moving {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
        #version 460

        layout(location = 0) in vec3 color;

        layout(location = 0) out vec4 f_color;

        void main() {
            f_color = vec4(color, 1.0);
        }
        "
    }
}
