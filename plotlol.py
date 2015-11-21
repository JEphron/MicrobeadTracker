import matplotlib.pyplot as plt
import matplotlib.animation as animation

# some utilities

class GeneratorPlotter():
    """Plots the output of a generator (with autoscaling)"""
    def __init__(self, title, axes, generator):
        self.fig, self.ax = plt.subplots()
        self.fig.suptitle(title, fontsize=20)
        plt.xlabel(axes['x'], fontsize=18)
        plt.ylabel(axes['y'], fontsize=16)
        line, = ax.plot([], [], lw=2)
        ax.grid()
        xdata, ydata = [], []
        
        global_ymin = 0
        global_ymax = 1

        def init():
            ax.set_ylim(0, 500)
            ax.set_xlim(0, len(img_sequence)/args.fps)
            del xdata[:]
            del ydata[:]
            line.set_data(xdata, ydata)
            return line,

        def update(data):
            # update the data
            t, y = data

            xdata.append(t)
            ydata.append(y)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            # print ymax, max(ydata), ymin, min(ydata)
            ax.set_ylim(min(ydata)-1, max(ydata)+1)
            ax.figure.canvas.draw()

            if t >= xmax:
                ax.set_xlim(xmin, 2 * xmax)
                ax.figure.canvas.draw()

            line.set_data(xdata, ydata)

            return line,

        def gen_area():
            # time axis for the plot
            t = 0
        
            # calculate the mean area of contours in the final frame
            average_area = 0
            final_frame_contours = get_contours(img_sequence[-1])
            for i in final_frame_contours:
                average_area = average_area + cv2.arcLength(i, True)
            average_area = average_area / len(final_frame_contours)

            min_area = sys.maxint
            for frame in process_frames(unprocessed_frames):
                if args.show:
                    cv2.imshow('window', frame)

                contours = get_contours(frame) # note: modifies source image
                show_contours(contours, frame, (0, 255, 0))

                area = 0
                average_area = 0
                
                for i in contours:
                    area = area + cv2.arcLength(i, True)
                
                if average_area is 0:
                    average_area = area / len(contours)
                
                if area / average_area < min_area:
                    min_area = area / average_area
                    print min_area

                # todo: don't do this
                if min_area <  10:
                    break
                
                t = t + 1
                # plt.savefig("out_fig/{num:03d}.png".format(num=t))
                yield t/args.fps, area

            # done with the data
            # todo: move this part somewhere else

            if args.png:
                plt.savefig(args.png)
                print "saved graph to {}".format(args.png)
            
            if args.csv:
                with open(args.csv, 'wb') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    for x, y in zip(xdata, ydata):
                        csvwriter.writerow([x, round(y)])
                print "saved csv to {}".format(args.csv)
                    


        # frame_generator = functools.partial(process_frames, unprocessed_frames)

        ani = animation.FuncAnimation(fig, update, gen_area, blit=False, interval=100,
                                      repeat=False, init_func=init)
        plt.show()