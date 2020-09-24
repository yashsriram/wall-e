# wall\_e

## description
- A simple manually controlled differential drive robot.

## roadmap
- Simulator.
    - [x] Interactive differential drive model.
    - [ ] Constraints on controls.
    - [ ] Wall boundaries.
- Controller impl.
    - [x] Conv, Relu needed?
        - Probably not. Since for the input (x, y, th, xg, yg) probably there no local relations or sequential memory required.
        - A simple strategy would be to orient towards goal. True theta can be obtained by a function of (x, y, xg, yg). w and v can be given to reduce theta reside and distance residue.
    - [x] Variable number of dynamic dof layers.
    - [x] Per layer activations.
    - [x] Random param initialization.
    - [x] Make ceo() a struct.
    - [x] Seperate bin crate for each network.
    - Speedup
        - [x] Parallelize.
        - [ ] Maybe move generation logic inside model? Removes into shapes a lot that way.
        - [ ] Use 8 core machine for training.
    - [x] sin().
    - [x] exp().
    - [ ] median().
    - [ ] Regularization.
- Controller design.
    - Model.
        - [ ] Input design.
        - [ ] Param init design.
        - [ ] Output design.
        - [ ] Hidden layer design.
    - Loss function.
        - [ ] Different loss functions.
        - [ ] Step level optimization vs Trajectory level optimization.
    - Optimizor.
        - [x] CEO.
- [ ] Report.

## code
- The code is written in stable rust.
- The structure is that of a typical rust project.
- `ggez` crate is used for rendering and event handling.

## documentation
- The documentation of the code is itself.

## usage
- Use `cargo run --release` to run the simulator.

## demonstration

