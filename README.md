## roadmap
- [x] Conv, Relu needed?
    - Probably not. Since for the input (x, y, th, xg, yg) probably there no local relations or sequential memory required.
    - A simple strategy would be to orient towards goal. True theta can be obtained by a function of (x, y, xg, yg). w and v can be given to reduce theta reside and distance residue.
- [x] Variable number of dynamic dof layers.
- [x] Per layer activations.
- [ ] Regularization.
- [ ] Maybe move generation logic inside model? Removes into shapes a lot that way.
- [ ] Parallelize.

- [ ] Input design.
- [ ] Output design.
- [ ] Hidden layer design.
