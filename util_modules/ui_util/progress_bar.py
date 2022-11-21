from tqdm.notebook import tqdm
class progress_bar:
    def __init__(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.display = False

    def display_bar(self):
        self.bar = tqdm(total=self.total_epoch)
        self.display = True
    def update_bar(self):
        if self.display == False:
            print("bar is not displayed yet!")
        else:
            self.bar.update(1)
            self.current_epoch += 1
            if self.current_epoch >= self.total_epoch:
                self.bar.close()            
    def clean_bar(self):
        self.bar.close()