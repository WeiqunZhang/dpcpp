When `assert(0)` is called on device, a message like below will be
printed out,

```
main.cpp:15: auto main(int, char **)::(anonymous class)::operator()(sycl::handler &)::(anonymous class)::operator()() const: global id: [0,0,0], local id: [0,0,0] Assertion `0` failed.
```

However, the program does not abort.  We can still see the message,

```
If you see this message, assert(0) did not abort.
```
