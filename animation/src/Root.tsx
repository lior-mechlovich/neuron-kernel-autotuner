import {Composition} from "remotion";
import {Workflow} from "./Workflow";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Workflow"
      component={Workflow}
      durationInFrames={540}
      fps={30}
      width={1280}
      height={720}
    />
  );
};
