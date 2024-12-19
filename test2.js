import {mat4, utils} from "wgpu-matrix";

const loc_position = 3;
const group_uniforms = 0;
const binding_uniforms = 2;

function init_quad(device) {
    const vertices = new Float32Array([
        -.5, -.5,
         .5, -.5,
         .5,  .5,
        -.5,  .5,
    ]);
    const vertexBuffer = device.createBuffer({
        label:"quad vertices",
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertexBuffer, 0, vertices);


    const indices = new Int16Array([
        0, 1, 2,
        0, 2, 3
    ]);
    const indexBuffer = device.createBuffer({
        label:"quad indices",
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indexBuffer, 0, indices);

    return {vertexBuffer, indexBuffer, count:6};
}

function init_pipeline(device, colorFormat) {
    const shaderModule = device.createShaderModule({
        label: "solid triangle shader",
        code:`
        struct Uniforms {
            color: vec3f,
            MVP: mat4x4f
        };

        @group(${group_uniforms}) @binding(${binding_uniforms}) var<uniform> uniforms: Uniforms;

        @vertex fn main_vert(@location(${loc_position}) position: vec2f)
            -> @builtin(position) vec4f {
            return uniforms.MVP * vec4f(position, 0, 1);
        }
        @fragment fn main_frag() -> @location(0) vec4f {
            return vec4f(uniforms.color, 1);
        }`,
 
    });
    const pipeline = device.createRenderPipeline({
        label: "solid triangle pipeline",
        layout: "auto",
        vertex: {
            module:shaderModule,
            entryPoint: "main_vert",
            buffers: [
                        {
                            arrayStride: 8,
                            attributes: [{
                                format: "float32x2",
                                offset: 0,
                                shaderLocation: loc_position,
                            }],

                        }
             ],
        },
        fragment: {
            module:shaderModule,
            entryPoint: "main_frag",
            targets: [{
                format: colorFormat,
            }]
        },
    });

    return pipeline;

}

function init_part(device, pipeline, color) {
    const uniforms = new Float32Array(4 + 4*4);
    uniforms.set(color, 0);

    const uniformBuffer = device.createBuffer({
        size: 4*(4 + 4*4), // vec3f + mat4x4f
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(group_uniforms),
        entries:[
            {binding: binding_uniforms, resource: {buffer: uniformBuffer}},
        ]
    });
    return {uniforms, uniformBuffer, bindGroup};
}


function setup_part(device, renderPass, MVP, part) {
    part.uniforms.set(MVP, 4);
    device.queue.writeBuffer(part.uniformBuffer, 0, part.uniforms);
    renderPass.setBindGroup(group_uniforms, part.bindGroup);
}

async function main() {
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    if(!device) {
        throw Error("WebGPU not supported.");
    }
    const canvas = document.querySelector("#webgpu");
    const context = canvas.getContext("webgpu");
    const preferredFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format: preferredFormat,
    });

    const quad = init_quad(device);
    const pipeline = init_pipeline(device, preferredFormat);
    const red_arm = init_part(device, pipeline, [1,0,0]);
    const green_arm = init_part(device, pipeline, [0,1,0]);
    const blue_finger_1 = init_part(device, pipeline, [0,0,1]);
    const blue_finger_2 = init_part(device, pipeline, [0,0,1]);

    let MVP;
    let MatStack = [];

    function render() {
        const WIDTH_RED = 0.3;
        const WIDTH_GREEN = 0.2;
        const LENGTH_GREEN = 0.8;
        const WIDTH_BLUE = (WIDTH_GREEN / 2.0);
        const LENGTH_BLUE = 0.3;

        let x_R = document.getElementById("x_R").value / 100.0;
        let y_R = document.getElementById("y_R").value / 100.0;
        let length_R = document.getElementById("length_R").value / 100.0;
        let angle_R = document.getElementById("angle_R").value;
        let length_G = document.getElementById("length_G").value / 100.0;
        let angle_G = document.getElementById("angle_G").value;
        let length_B = document.getElementById("length_B").value / 100.0;
        let angle_B = document.getElementById("angle_B").value;

        
        MVP = mat4.identity();
    /*
                     P
                     |
                     V
                     |
                  T_base
                     |
                     Rr
                    /  \
               Tr_high  Tr_low
                  /      |
                 Rg      Sr
                /|\      |
               / | \  red_arm 
              /  |  \   
             /   |   \      
            /    |    \
       Tg_low Tg_high1 Tg_high2
          |      |      |
          Sg    Rb1    Rb2
          |      |      | 
     green_arm   Tb     Tb
                 |      |
                 Sb     Sb
                 |      |
      blue_finger_1    blue_finger_2
    */
        const encoder = device.createCommandEncoder();
        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                clearValue: {r:0, g:0, b:0.4, a:1},
                storeOp: "store",
            }]
        });
        renderPass.setPipeline(pipeline);
        renderPass.setVertexBuffer(0, quad.vertexBuffer);
        renderPass.setIndexBuffer(quad.indexBuffer, 'uint16');
 
        MVP = mat4.ortho(-2,2,-2,2,0,4);    // P
        MVP = mat4.translate(MVP, [0,0,-2]);    // V

        MVP = mat4.translate(MVP, [x_R, y_R, 0]);   // T_base
        MVP = mat4.rotate(MVP, [0,0,1], utils.degToRad(angle_R));   // Rr

        MatStack.push(mat4.clone(MVP));
            MVP = mat4.translate(MVP, [(length_R-WIDTH_RED)/2.0, 0, 0.1]);  // Tr_low
            MVP = mat4.scale(MVP, [length_R, WIDTH_RED, 1]);    // Sr
            setup_part(device, renderPass, MVP, red_arm);
            renderPass.drawIndexed(quad.count);
        MVP = MatStack.pop();

        MatStack.push(mat4.clone(MVP));
            MVP = mat4.translate(MVP, [length_R - WIDTH_RED, 0, 0]); // Tr_high
            MVP = mat4.rotate(MVP, [0,0,1], utils.degToRad(angle_G)); // Rg
    
            MatStack.push(mat4.clone(MVP));
                MVP = mat4.translate(MVP, [(length_G - WIDTH_GREEN)/2.0, 0, 0.3]); // Tg_low
                MVP = mat4.scale(MVP, [length_G, WIDTH_GREEN, 1]); // Sg
                setup_part(device, renderPass, MVP, green_arm);
                renderPass.drawIndexed(quad.count);
            MVP = MatStack.pop();
    
            MatStack.push(mat4.clone(MVP));
                MVP = mat4.translate(MVP, [length_G - WIDTH_GREEN + WIDTH_BLUE/2.0, WIDTH_BLUE/2.0, 0]); // Tg_hight1
                MVP = mat4.rotate(MVP, [0,0,1], utils.degToRad(angle_B)); // Rb1
                MVP = mat4.translate(MVP, [(length_B - WIDTH_BLUE)/2.0, 0, 0.3]); // Tb
                MVP = mat4.scale(MVP, [length_B, WIDTH_BLUE, 1]); // Sb
                setup_part(device, renderPass, MVP, blue_finger_1);
                renderPass.drawIndexed(quad.count);
            MVP = MatStack.pop();
    
            MatStack.push(mat4.clone(MVP));
                MVP = mat4.translate(MVP, [length_G - WIDTH_GREEN + WIDTH_BLUE/2.0, -WIDTH_BLUE/2.0, 0]); // Tg_hight2
                MVP = mat4.rotate(MVP, [0,0,1], utils.degToRad(-angle_B)); // Rb2
                MVP = mat4.translate(MVP, [(length_B - WIDTH_BLUE)/2.0, 0, 0.3]); // Tb
                MVP = mat4.scale(MVP, [length_B, WIDTH_BLUE, 1]); // Sb
                setup_part(device, renderPass, MVP, blue_finger_2);
                renderPass.drawIndexed(quad.count);
            MVP = MatStack.pop();
        MVP = MatStack.pop();

   
        renderPass.end();
        const commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);

        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}



main();
