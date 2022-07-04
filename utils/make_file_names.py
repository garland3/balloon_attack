from pathlib import Path


class Make_file_names:
    def __init__(self, args):
        self.args = args
        self.output_folder = Path(args.out_path)
        self.output_folder.mkdir(exist_ok=True)

    def make_base_name(self, i, name_prefix=None):
        name = f"{name_prefix}_" if name_prefix else ""
        self.base_name = str(
            self.output_folder / f"{name}{i}_{self.args.weights}_webcam"
        )
        return self.base_name

    def output_image(self, i, data):
        filename = self.make_base_name(i)
        if self.args.bbox:
            filename += "_bbox"
        if self.args.pose:
            filename += "_pose"
        if self.args.face:
            filename += "_face"
        if self.args.kp_bbox:
            filename += "_kpbbox"
        if data["use_kp_dets"]:
            filename += "_kp_obj"
        filename += ".png"
        return filename

    def poses_name(self, i):
        filename = self.make_base_name(i, name_prefix="poses")
        # filename += ".json"
        return filename
