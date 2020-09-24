## roadmap
### impl
- [x] Conv, Relu needed?
    - Probably not. Since for the input (x, y, th, xg, yg) probably there no local relations or sequential memory required.
    - A simple strategy would be to orient towards goal. True theta can be obtained by a function of (x, y, xg, yg). w and v can be given to reduce theta reside and distance residue.
- [x] Variable number of dynamic dof layers.
- [x] Per layer activations.
- [x] Random param initialization.
- [x] Make ceo() a struct.
- [x] Parallelize.
- [x] Seperate bin crate for each network.
- [ ] Maybe move generation logic inside model? Removes into shapes a lot that way.
- [ ] Use 8 core machine for training.

### task specific design
- [ ] Input design.
- [ ] Param init design.
- [ ] Output design.
- [ ] Hidden layer design.
- [ ] Other loss functions.
- [ ] Regularization.
