struct Device{Device()=default;Device(const Device&)=delete;};
int main(){Device a;Device b(a);}
